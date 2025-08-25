# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Union
from warnings import warn

from torchtune.modules.transforms import Transform

Role = Literal[
    "system",  # Origin is system prompt
    "user",  # Origin is user
    "assistant",  # Origin is the model output
    "ipython",  # Origin is return from a tool call
    "tool",  # Origin is return from a tool call
]


class MaskingStrategy(Enum):
    TRAIN_ON_ALL = "train_on_all"
    TRAIN_ON_ASSISTANT = "train_on_assistant"
    TRAIN_ON_LAST = "train_on_last"


class Message:
    """
    This class represents individual messages in a fine-tuning dataset. It supports
    text-only content, text with interleaved images, and tool calls. The
    :class:`~torchtune.modules.transforms.tokenizers.ModelTokenizer` will tokenize
    the content of the message using ``tokenize_messages`` and attach the appropriate
    special tokens based on the flags set in this class.

    Args:
        role (Role): role of the message writer. Can be "system" for system prompts,
            "user" for human prompts, "assistant" for model responses, or "ipython"
            for tool call returns.
        content (Union[str, list[dict[str, Any]]]): content of the message. If it is text only content,
            you can pass in a string. If it is multimodal content, pass in a list of dictionaries formatted
            as follows::

                [
                    {"type": "image", "content": torch.Tensor},
                    {"type": "text", "content": "What is in this image?"},
                ]

        masked (bool): whether the message is masked in the sample. If True, do not use
            in loss calculation. Default: False
        tool (bool): whether the message is a tool call. Default: False
        tool_calls (Optional[list]): list of tool calls related to this message. Default: None
        eot (bool): whether the message corresponds to the end of a turn, where control is handed over
            to the assistant from the user or the user from the assistant. Default: True. Should be true
            in most cases except for:

            - For multiple consecutive assistant messages (i.e., tool calls
              by assistant), only the last assistant message will have ``eot=True``
            - All tool messages (tool call returns) should set ``eot=False``.

    Note:
        Message class expects any image content to be a ``torch.Tensor``, as output
        by e.g. :func:`~torchtune.data.load_image`
    """

    def __init__(
        self,
        role: Role,
        content: Union[str, list[dict[str, Any]]],
        masked: bool = False,
        tool: bool = False,
        tool_calls: Optional[list] = None,
        eot: bool = True,
    ):
        self.role = role
        self.content = self._convert_to_list_of_dict(content)
        self.masked = masked
        self.tool = tool
        self.tool_calls = tool_calls
        self.eot = eot

        self._validate_message()

    def _convert_to_list_of_dict(self, content) -> list[dict[str, Any]]:
        """User is currently allowed to pass in a string for text-only content.
        This ensures that the content is formatted as a list of dictionaries."""
        if isinstance(content, str):
            return [{"type": "text", "content": content}]

        assert isinstance(
            content, list
        ), f"content must be of type list[dict[str, Any]], got {content}"

        return content

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        """
        Construct a Message from a dictionary.

        Args:
            d (dict): dictionary containing the fields of the Message.

        Returns:
            Message: constructed Message.
        """
        return cls(
            role=d["role"],
            content=d["content"],
            masked=d.get("masked", False),
            tool_calls=d.get("tool_calls", []),
            tool=d.get("tool", False),
            eot=d.get("eot", True),
        )

    def get_media(self) -> list["PIL.Image.Image"]:
        """
        Returns media content of the message.
        """
        return [
            content["content"] for content in self.content if content["type"] == "image"
        ]

    @property
    def contains_media(self) -> bool:
        """
        Returns whether the message contains media.
        """
        return any(content["type"] == "image" for content in self.content)

    @property
    def text_content(self) -> str:
        """
        Returns text-only content of the message.
        """
        return "".join(
            content["content"] for content in self.content if content["type"] == "text"
        )

    def _validate_message(self) -> None:
        # Sometime existence of tool call is checked via self.tool_calls without self.tool.
        if self.tool or self.tool_calls and self.contains_media:
            raise ValueError(
                f"Media tokens in tool calls are not supported. Both are set in message: {self.text_content}"
            )
        if self.tool or self.tool_calls and self.role != "assistant":
            raise ValueError(
                f"Only assistant messages can be tool calls. Found role {self.role} in message: {self.text_content}"
            )

    def __repr__(self) -> str:
        content_only = [content["content"] for content in self.content]
        return f"Message(role='{self.role}', content={content_only!r})"


class InputOutputToMessages(Transform):
    """
    Message transform class that converts a single sample with "input" and "output" fields,
    (or equivalent fields specified in column_map) to user and assistant messages,
    respectively. This is useful for datasets that have two columns, one containing
    the user prompt string and the other containing the model response string::

        |  input          |  output          |
        |-----------------|------------------|
        | "user prompt"   | "model response" |

    Args:
        train_on_input (Optional[bool]): whether the model is trained on the user prompt or not.
            Deprecated parameter and will be removed in a future release.
            Default is None.
        column_map (Optional[dict[str, str]]): a mapping to change the expected "input"
            and "output" column names to the actual column names in the dataset. Keys should
            be "input" and "output" and values should be the actual column names. Default is None,
            keeping the default "input" and "output" column names.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Default is None.
        image_dir (Optional[Path]): path to the directory containing the images that is prepended to all image
            paths in the dataset. For example, if ``image_dir="/home/user/dataset/"` and the sample image path
            was ``"images/1.jpg"``, the final image path that will be loaded is ``"/home/user/dataset/images/1.jpg"``.
            If None, assume images are available in current working directory or are located
            on a remote url. For text-only, leave as None. Default is None.
        masking_strategy (Optional[str]): masking strategy to use for model training.
            Must be one of: `train_on_all`, `train_on_assistant`, `train_on_last`.
            Default is "train_on_assistant".

            - ``train_on_all``: both user and assistant messages are unmasked
            - ``train_on_assistant``: user messages are masked, only assistant messages are unmasked
            - ``train_on_last``: only the last assistant message is unmasked

            Note: Multimodal user messages are always masked.

    Raises:
        ValueError:
            If ``column_map`` is provided and ``input`` not in ``column_map``, or
                ``output`` not in ``column_map``, **or**
            if ``image_dir`` is provided but ``image`` not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: Optional[bool] = None,
        column_map: Optional[dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
        image_dir: Optional[Path] = None,
        masking_strategy: Optional[str] = "train_on_assistant",
    ):
        if train_on_input is not None:
            warn(
                "train_on_input is deprecated and will be removed in a future release. "
                "Please use masking_strategy instead."
                "You should replace train_on_input=True with masking_strategy='train_on_all', and "
                "train_on_input=False with masking_strategy='train_on_assistant'."
                "For backwards compatibility, if you pass both train_on_input and masking_strategy, "
                "the value of masking_strategy will be ignored until torchtune 0.7. ",
                DeprecationWarning,
                stacklevel=2,
            )
            masking_strategy = (
                "train_on_all" if train_on_input else "train_on_assistant"
            )
        self.masking_strategy = masking_strategy
        self.new_system_prompt = new_system_prompt

        if column_map:
            if "input" not in column_map:
                raise ValueError(
                    f"Expected a key of 'input' in column_map but found {column_map.keys()}."
                )
            if "output" not in column_map:
                raise ValueError(
                    f"Expected a key of 'output' in column_map but found {column_map.keys()}."
                )
            self._column_map = column_map
        else:
            self._column_map = {"input": "input", "output": "output", "image": "image"}

        # Ensure that if a user seems to want to construct a multimodal transform, they provide
        # a proper column_mapping
        if "image" not in self._column_map.keys() and image_dir is not None:
            raise ValueError(
                f"image_dir is specified as {image_dir} but 'image' is not in column_map. "
                "Please specify an 'image' key in column_map."
            )

        self.image_dir = image_dir

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        from torchtune.data._utils import load_image

        is_multimodal = "image" in sample or (
            "image" in self._column_map and self._column_map["image"] in sample
        )

        if is_multimodal:
            image_path = sample[self._column_map["image"]]
            if isinstance(image_path, str):
                # Convert image_path to Path obj
                image_path = Path(image_path)

                # If image_dir is not None, prepend image_dir to image_path
                if self.image_dir is not None:
                    image_path = self.image_dir / image_path

                # Load if not loaded
                pil_image = load_image(image_path)
            else:
                pil_image = image_path
            content = [
                {"type": "image", "content": pil_image},
                {"type": "text", "content": sample[self._column_map["input"]]},
            ]
        else:
            content = [{"type": "text", "content": sample[self._column_map["input"]]}]

        output_content = [
            {"type": "text", "content": sample[self._column_map["output"]]}
        ]

        messages = [
            Message(
                role="user",
                content=content,
                eot=True,
            ),
            Message(
                role="assistant",
                content=output_content,
                eot=True,
            ),
        ]
        if self.new_system_prompt is not None:
            messages = [
                Message(
                    role="system", content=self.new_system_prompt, masked=True, eot=True
                )
            ] + messages
        mask_messages(messages, self.masking_strategy)
        return {"messages": messages}


class ChosenRejectedToMessages(Transform):
    """
    Transform for converting a single sample from datasets with "chosen" and "rejected" columns
    containing conversations to a list of chosen and rejected messages. For example::

        |  chosen                                |  rejected                              |
        |----------------------------------------|----------------------------------------|
        | [{"role": "user", "content": Q1},      | [{"role": "user", "content": Q1},      |
        |  {"role": "assistant", "content": A1}] |  {"role": "assistant", "content": A2}] |

    will be converted to:

    .. code-block:: python

        chosen = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]
        rejected = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A2"),
        ]

    A single sample typically consists of a single optional system prompt and one or multiple
    turns of user and assistant messages.

    Args:
        train_on_input (Optional[bool]): whether the model is trained on the user prompt or not.
            Deprecated parameter and will be removed in a future release.
            Default is None.
        column_map (Optional[dict[str, str]]): a mapping to change the expected
            "chosen" and "rejected" column names to the actual column names in the dataset.
            Keys should be "chosen" and "rejected" and values should be the actual column names.
            Default is None, keeping the default column names.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.
        masking_strategy (Optional[str]): masking strategy to use for model training.
            Must be one of: `train_on_all`, `train_on_assistant`, `train_on_last`.
            Default is "train_on_assistant".

            - ``train_on_all``: both user and assistant messages are unmasked
            - ``train_on_assistant``: user messages are masked, only assistant messages are unmasked
            - ``train_on_last``: only the last assistant message is unmasked

    Raises:
        ValueError: If ``column_map`` is provided and ``chosen`` not in ``column_map``, or
            ``rejected`` not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: Optional[bool] = None,
        column_map: Optional[dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
        masking_strategy: Optional[str] = "train_on_assistant",
    ):
        if train_on_input is not None:
            warn(
                "train_on_input is deprecated and will be removed in a future release. "
                "Please use masking_strategy instead."
                "You should replace train_on_input=True with masking_strategy='train_on_all', and "
                "train_on_input=False with masking_strategy='train_on_assistant'."
                "For backwards compatibility, if you pass both train_on_input and masking_strategy, "
                "the value of masking_strategy will be ignored until torchtune 0.7. ",
                DeprecationWarning,
                stacklevel=2,
            )
            masking_strategy = (
                "train_on_all" if train_on_input else "train_on_assistant"
            )
        self.masking_strategy = masking_strategy
        self.new_system_prompt = new_system_prompt
        if column_map:
            if "chosen" not in column_map:
                raise ValueError(
                    f"Expected a key of 'chosen' in column_map but found {column_map.keys()}."
                )
            if "rejected" not in column_map:
                raise ValueError(
                    f"Expected a key of 'rejected' in column_map but found {column_map.keys()}."
                )
            self._column_map = column_map
        else:
            self._column_map = {"chosen": "chosen", "rejected": "rejected"}

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        chosen_messages = []
        for message in sample[self._column_map["chosen"]]:
            if message["role"] == "system" and self.new_system_prompt is not None:
                continue
            chosen_messages.append(Message.from_dict(message))

        rejected_messages = []
        for message in sample[self._column_map["rejected"]]:
            if message["role"] == "system" and self.new_system_prompt is not None:
                continue
            rejected_messages.append(Message.from_dict(message))

        if self.new_system_prompt is not None:
            chosen_messages = [
                Message(
                    role="system", content=self.new_system_prompt, masked=True, eot=True
                )
            ] + chosen_messages
            rejected_messages = [
                Message(
                    role="system", content=self.new_system_prompt, masked=True, eot=True
                )
            ] + rejected_messages
        mask_messages(chosen_messages, self.masking_strategy)
        mask_messages(rejected_messages, self.masking_strategy)
        return {"chosen": chosen_messages, "rejected": rejected_messages}


class ShareGPTToMessages(Transform):
    """
    Convert a single chat sample adhering to the ShareGPT JSON structure to torchtune's :class:`~torchtune.data.Message`
    structure.

    A single sample typically consists of a single optional system prompt and one or multiple
    turns of user and assistant messages.

    ShareGPT follows::

        {
            "conversations": [
                {
                    "from": <system|human|gpt>,
                    "value": <message>,
                },
                ...
            ]
        }

    :class:`~torchtune.data.Message` follows::

        [
            {
                "role": <system|user|assistant>,
                "content": <message>,
            },
            ...
        ]

    Args:
        train_on_input (Optional[bool]): whether the model is trained on the user prompt or not.
            Deprecated parameter and will be removed in a future release.
            Default is None.
        column_map (Optional[dict[str, str]]): a mapping from the expected columns ("conversations")
            to the new column names in the dataset. Key should be "conversations" and value should
            be the new column name. If None, keep the default "conversations".
            Default is None.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.
        image_dir (Optional[Path]): path to the directory containing the images that is prepended to all image
            paths in the dataset. For example, if ``image_dir="/home/user/dataset/"` and the sample image path
            was ``"images/1.jpg"``, the final image path that will be loaded is ``"/home/user/dataset/images/1.jpg"``.
            If None, assume images are available in current working directory or are located
            on a remote url. For text-only, leave as None. Default is None.
        image_tag (Optional[str]): placeholder tags in the text content of each message to be replaced by image
            special tokens. If images are present and this is None, then will prepend image tokens to the first
            user message in the sample by default. If text-only, this field is ignored. Default is ``"<image>"``.
        masking_strategy (Optional[str]): masking strategy to use for model training.
            Must be one of: `train_on_all`, `train_on_assistant`, `train_on_last`.
            Default is "train_on_assistant".

            - ``train_on_all``: both user and assistant messages are unmasked
            - ``train_on_assistant``: user messages are masked, only assistant messages are unmasked
            - ``train_on_last``: only the last assistant message is unmasked

            Note: Multimodal user messages are always masked.

    Raises:
        ValueError: If ``column_map`` is provided and ``conversations`` not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: Optional[bool] = None,
        column_map: Optional[dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
        image_dir: Optional[Path] = None,
        image_tag: Optional[str] = "<image>",
        masking_strategy: Optional[str] = "train_on_assistant",
    ):
        if train_on_input is not None:
            warn(
                "train_on_input is deprecated and will be removed in a future release. "
                "Please use masking_strategy instead."
                "You should replace train_on_input=True with masking_strategy='train_on_all', and "
                "train_on_input=False with masking_strategy='train_on_assistant'."
                "For backwards compatibility, if you pass both train_on_input and masking_strategy, "
                "the value of masking_strategy will be ignored until torchtune 0.7. ",
                DeprecationWarning,
                stacklevel=2,
            )
            masking_strategy = (
                "train_on_all" if train_on_input else "train_on_assistant"
            )
        self.masking_strategy = masking_strategy
        self.new_system_prompt = new_system_prompt
        if column_map:
            if "conversations" not in column_map:
                raise ValueError(
                    f"Expected a key of 'conversations' in column_map but found {column_map.keys()}."
                )
            self._column_map = column_map
        else:
            self._column_map = {"conversations": "conversations", "image": "image"}
        self.image_dir = image_dir
        self.image_tag = image_tag

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Return a list of Message objects from the provided sample dict.

        Args:
            sample (Mapping[str, Any]): a single data sample with "messages" field pointing
                to a list of dict messages.

        Returns:
            list[Message]: A list of messages with "role" and "content" fields.
        """
        from torchtune.data._utils import format_content_with_images, load_image

        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        messages = []
        if self.new_system_prompt is not None:
            messages.append(
                Message(
                    role="system", content=self.new_system_prompt, masked=True, eot=True
                )
            )

        is_multimodal = "image" in sample or (
            "image" in self._column_map and self._column_map["image"] in sample
        )

        # Gate variable to ensure that we only prepend image tokens to the first user message
        image_loaded = False
        for message in sample[self._column_map["conversations"]]:
            role = role_map[message["from"]]
            content = message["value"]
            if role == "system" and self.new_system_prompt is not None:
                continue
            if role == "user":
                if is_multimodal and not image_loaded:
                    image_path = sample[self._column_map["image"]]
                    if self.image_dir is not None:
                        image_path = self.image_dir / image_path
                    pil_image = load_image(image_path)
                    # If image tag is not specified, prepend by default
                    if self.image_tag is None:
                        content = [
                            {"type": "image", "content": pil_image},
                            {"type": "text", "content": content},
                        ]
                    else:
                        content = format_content_with_images(
                            content,
                            image_tag=self.image_tag,
                            images=[pil_image],
                        )
                    image_loaded = True
            messages.append(Message(role=role, content=content))
        mask_messages(messages, self.masking_strategy)
        return {"messages": messages}


class OpenAIToMessages(Transform):
    """
    Convert a single chat sample adhering to the `OpenAI chat completion <https://platform.openai.com/docs/api-reference/chat>`_
    JSON structure to torchtune's :class:`~torchtune.data.Message` structure. This supports both
    text and image messages.

    A single sample typically consists of a single optional system prompt and one or multiple
    turns of user and assistant messages.

    For example::

        {
            "messages": [
                {
                    "role": <system|user|assistant>,
                    "content": [
                        {
                            "type": "text",
                            "text": "What'\''s in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": <url>,
                            },
                        },
                },
                ...
            ]
        }

    :class:`~torchtune.data.Message` follows::

        [
            {
                "role": <system|user|assistant>,
                "content": [
                    {
                        "type": "text",
                        "content": "What'\''s in this image?",
                    },
                    {
                        "type": "image",
                        "content": <PIL.Image.Image>,
                    },
                ],
            },
            ...
        ]

    Args:
        train_on_input (Optional[bool]): whether the model is trained on the user prompt or not.
            Deprecated parameter and will be removed in a future release.
            Default is None.
        column_map (Optional[dict[str, str]]): a mapping from the expected columns ("messages")
            to the new column names in the dataset. Key should be "messages" and value should be
            the new column name. If None, keep the default "messages".
            Default is None.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.
        masking_strategy (Optional[str]): masking strategy to use for model training.
            Must be one of: `train_on_all`, `train_on_assistant`, `train_on_last`.
            Default is "train_on_assistant".

            - ``train_on_all``: both user and assistant messages are unmasked
            - ``train_on_assistant``: user messages are masked, only assistant messages are unmasked
            - ``train_on_last``: only the last assistant message is unmasked

            Note: Multimodal user messages are always masked.

    Raises:
        ValueError: If ``column_map`` is provided and ``messages`` not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: Optional[bool] = None,
        column_map: Optional[dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
        masking_strategy: Optional[str] = "train_on_assistant",
    ):
        if train_on_input is not None:
            warn(
                "train_on_input is deprecated and will be removed in a future release. "
                "Please use masking_strategy instead."
                "You should replace train_on_input=True with masking_strategy='train_on_all', and "
                "train_on_input=False with masking_strategy='train_on_assistant'."
                "For backwards compatibility, if you pass both train_on_input and masking_strategy, "
                "the value of masking_strategy will be ignored until torchtune 0.7. ",
                DeprecationWarning,
                stacklevel=2,
            )
            masking_strategy = (
                "train_on_all" if train_on_input else "train_on_assistant"
            )
        self.masking_strategy = masking_strategy
        self.new_system_prompt = new_system_prompt
        if column_map:
            if "messages" not in column_map:
                raise ValueError(
                    f"Expected a key of 'messages' in column_map but found {column_map.keys()}."
                )
            self._column_map = column_map
        else:
            self._column_map = {"messages": "messages"}

    def _convert_from_openai_content(
        self, content: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Converts a list of content dicts from the OpenAI format to the torchtune format."""
        from torchtune.data._utils import load_image

        converted_content = []
        for content_dict in content:
            if content_dict["type"] == "text":
                converted_content.append(
                    {"type": "text", "content": content_dict["text"]}
                )
            elif content_dict["type"] == "image_url":
                converted_content.append(
                    {
                        "type": "image",
                        "content": load_image(content_dict["image_url"]["url"]),
                    }
                )
        return converted_content

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Return a list of Message objects from the provided sample dict.

        Args:
            sample (Mapping[str, Any]): a single data sample with "messages" field pointing
                to a list of dict messages.

        Returns:
            list[Message]: A list of messages with "role" and "content" fields.
        """
        updated_messages = []
        if self.new_system_prompt is not None:
            updated_messages.append(
                Message(
                    role="system", content=self.new_system_prompt, masked=True, eot=True
                )
            )
        messages = sample[self._column_map["messages"]]
        for i, message in enumerate(messages):
            if message["role"] == "system" and self.new_system_prompt is not None:
                continue
            if isinstance(message["content"], list):
                content = self._convert_from_openai_content(message["content"])
            elif isinstance(message["content"], str):
                content = message["content"]

            eot = True
            if message["role"] in ["tool", "ipython"]:
                # After tool responses, turn is not over, because assistant will interpret the tool response.
                eot = False
            elif message["role"] == "assistant":
                # If the next message is a tool response instead of a user message
                # the current assistant message is not the end of the turn.
                # Models like Llama will append EOM to the end of the assistant message for tool calls.
                has_next_message = i < len(messages) - 1
                if has_next_message and messages[i + 1]["role"] in ["tool", "ipython"]:
                    eot = False

            updated_messages.append(
                Message(
                    role=message["role"],
                    content=content,
                    eot=eot,
                ),
            )
        mask_messages(updated_messages, self.masking_strategy)
        return {"messages": updated_messages}


class AlpacaToMessages(Transform):
    """
    Message transform class for Alpaca-style datasets with "instruction", "input", and "output"
    (or equivalent fields specified in column_map) columns. User messages are formed from the
    instruction + input columns and assistant messages are formed from the output column. Prompt
    templating is conditional on the presence of the "input" column, and thus is handled directly
    in this transform class instead of a dedicated :class:`~torchtune.data.PromptTemplate` class
    due to this custom logic.

    Args:
        train_on_input (Optional[bool]): whether the model is trained on the user prompt or not.
            Deprecated parameter and will be removed in a future release.
            Default is None.
        column_map (Optional[dict[str, str]]): a mapping to change the expected "instruction", "input",
            and "output" column names to the actual column names in the dataset. Default is None,
            keeping the default column names.
        masking_strategy (Optional[str]): masking strategy to use for model training.
            Must be one of: `train_on_all`, `train_on_assistant`, `train_on_last`.
            Default is "train_on_all".

            - ``train_on_all``: both user and assistant messages are unmasked
            - ``train_on_assistant``: user messages are masked, only assistant messages are unmasked
            - ``train_on_last``: only the last assistant message is unmasked

    Raises:
        ValueError:
            If ``column_map`` is provided and ``instruction`` not in ``column_map``, or
                ``output`` not in ``column_map``
    """

    def __init__(
        self,
        train_on_input: Optional[bool] = None,
        column_map: Optional[dict[str, str]] = None,
        masking_strategy: Optional[str] = "train_on_all",
    ):
        if train_on_input is not None:
            warn(
                "train_on_input is deprecated and will be removed in a future release. "
                "Please use masking_strategy instead."
                "You should replace train_on_input=True with masking_strategy='train_on_all', and "
                "train_on_input=False with masking_strategy='train_on_assistant'."
                "For backwards compatibility, if you pass both train_on_input and masking_strategy, "
                "the value of masking_strategy will be ignored until torchtune 0.7. ",
                DeprecationWarning,
                stacklevel=2,
            )
            masking_strategy = (
                "train_on_all" if train_on_input else "train_on_assistant"
            )
        self.masking_strategy = masking_strategy
        if column_map:
            if "instruction" not in column_map:
                raise ValueError(
                    f"Expected a key of 'instruction' in column_map but found {column_map.keys()}."
                )
            # input is optional
            if "output" not in column_map:
                raise ValueError(
                    f"Expected a key of 'output' in column_map but found {column_map.keys()}."
                )
            self._column_map = column_map
        else:
            self._column_map = {
                "instruction": "instruction",
                "input": "input",
                "output": "output",
            }
        self.template = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            ),
        }

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        key_input = self._column_map.get("input", "input")
        if key_input in sample and sample[key_input]:
            prompt = self.template["prompt_input"].format(
                instruction=sample[self._column_map["instruction"]],
                input=sample[key_input],
            )
        else:
            prompt = self.template["prompt_no_input"].format(
                instruction=sample[self._column_map["instruction"]]
            )

        messages = [
            Message(
                role="user",
                content=prompt,
                eot=True,
            ),
            Message(
                role="assistant",
                content=sample[self._column_map["output"]],
                eot=True,
            ),
        ]
        mask_messages(messages, self.masking_strategy)
        return {"messages": messages}


def validate_messages(
    messages: list[Message],
) -> None:
    """
    Given a list of messages, ensure that messages form a valid
    back-and-forth conversation. An error will be raised if:

    - There is a system message that's not the first message
    - There are two consecutive user messages
    - An assistant message comes before the first user message
    - The message is empty
    - Messages are shorter than length of 2 (min. one user-assistant turn)


    Args:
        messages (list[Message]): the messages to validate.

    Raises:
        ValueError: If the messages are invalid.
    """
    if len(messages) < 2:
        raise ValueError(
            f"Messages must be at least length 2, but got {len(messages)} messages"
        )

    last_message = Message(role="assistant", content="")
    for i, message in enumerate(messages):
        if message.role == "assistant" and last_message.role not in [
            "user",
            "tool",
            "ipython",
        ]:
            raise ValueError(
                f"Assistant message before expected user, tool or ipython message at index {i} in messages"
            )
        if message.role == "user" and last_message.role == "user":
            raise ValueError(
                f"Two consecutive user messages at index {i} and {i - 1} in messages"
            )
        if message.role == "system" and i > 0:
            raise ValueError(
                f"System message at index {i} in messages, but system messages must come first"
            )
        if message.role in ["tool", "ipython"] and not last_message.ipython:
            raise ValueError(
                f"Tool or ipython message at index {i} must follow an ipython message"
            )
        last_message = message


def mask_messages(messages: list[Message], masking_strategy: MaskingStrategy) -> None:
    """
    Set the masked attribute for each message in the list based on the specified masking strategy.

    Args:
        messages (list[Message]): a list of messages to mask.
        masking_strategy (MaskingStrategy): masking strategy to use.
            Must be one of `train_on_all`, `train_on_assistant`, `train_on_last`.

            - ``train_on_all``: both user and assistant messages are unmasked
            - ``train_on_assistant``: user messages are masked, only assistant messages are unmasked
            - ``train_on_last``: only the last assistant message is unmasked
    """
    masking_strategy = MaskingStrategy(masking_strategy)
    marked_last_assistant_message = False
    for message in reversed(messages):
        # System messages are always masked
        if message.role == "system":
            message.masked = True
            continue
        if masking_strategy == MaskingStrategy.TRAIN_ON_LAST:
            if message.role == "assistant" and not marked_last_assistant_message:
                message.masked = False
                marked_last_assistant_message = True
            else:
                message.masked = True
        # Multimodal user messages are always masked
        elif masking_strategy == MaskingStrategy.TRAIN_ON_ALL:
            message.masked = message.role == "user" and message.contains_media
        elif masking_strategy == MaskingStrategy.TRAIN_ON_ASSISTANT:
            message.masked = message.role != "assistant"
