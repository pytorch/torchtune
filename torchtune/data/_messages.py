# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from torchtune.modules.transforms import Transform

Role = Literal[
    "system",  # Origin is system prompt
    "user",  # Origin is user
    "assistant",  # Origin is the model output
    "ipython",  # Origin is return from a tool call
]


class Message:
    """
    This class represents individual messages in a fine-tuning dataset. It supports
    text-only content, text with interleaved images, and tool calls. The :class:`~torchtune.modules.tokenizers.ModelTokenizer`
    will tokenize the content of the message using ``tokenize_messages`` and attach
    the appropriate special tokens based on the flags set in this class.

    Args:
        role (Role): role of the message writer. Can be "system" for system prompts,
            "user" for human prompts, "assistant" for model responses, or "ipython"
            for tool call returns.
        content (Union[str, List[Dict[str, str]]]): content of the message. If it is text only content,
            you can pass in a string. If it is multimodal content, pass in a list of dictionaries formatted
            as follows::

                [
                    {"type": "image"}
                    {"type": "text", "content": "hello"},
                    {"type": "image"}
                    {"type": "text", "content": "world"},
                ]

        masked (bool): whether the message is masked in the sample. If True, do not use
            in loss calculation. Default: False
        ipython (bool): whether the message is a tool call. Default: False
        eot (bool): whether the message corresponds to the end of a turn, where control is handed over
            to the assistant from the user or the user from the assistant. Default: True. Should be true
            in most cases except for:

            - For multiple consecutive assistant messages (i.e., tool calls
              by assistant), only the last assistant message will have ``eot=True``
            - All ipython messages (tool call returns) should set ``eot=False``.
    """

    def __init__(
        self,
        role: Role,
        content: Union[str, List[Dict[str, str]]],
        masked: bool = False,
        ipython: bool = False,
        eot: bool = True,
    ):
        self.role = role
        self.content = (
            [{"type": "text", "content": content}]
            if isinstance(content, str)
            else content
        )
        self.masked = masked
        self.ipython = ipython
        self.eot = eot

        self._validate_message()

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
            ipython=d.get("ipython", False),
            eot=d.get("eot", True),
        )

    @property
    def contains_media(self) -> bool:
        """
        Returns True if message contains non-text content.
        """
        return any(content["type"] != "text" for content in self.content)

    @property
    def text_content(self) -> str:
        """
        Returns text-only content of the message.
        """
        return "".join(
            content["content"] for content in self.content if content["type"] == "text"
        )

    def _validate_message(self) -> None:
        if self.ipython and self.contains_media:
            raise ValueError(
                f"Media tokens in tool calls are not supported. Both are set in message: {self.text_content}"
            )
        if self.ipython and self.role != "assistant":
            raise ValueError(
                f"Only assistant messages can be tool calls. Found role {self.role} in message: {self.text_content}"
            )


class InputOutputToMessages(Transform):
    """
    Message transform class that converts a single sample with "input" and "output" fields,
    (or equivalent fields specified in column_map) to user and assistant messages,
    respectively. This is useful for datasets that have two columns, one containing
    the user prompt and the other containing the model response.

    Args:
        train_on_input (bool): Whether the model is trained on the user prompt or not.
            Default is False.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "input"
            and "output" column names to the actual column names in the dataset. Default is None,
            keeping the default "input" and "output" column names.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Default is None.

    Raises:
        ValueError: If ``column_map`` is provided and ``input`` not in ``column_map``, or
            ``output`` not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: bool = False,
        column_map: Optional[Dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
    ):
        self.train_on_input = train_on_input
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
            self._column_map = {"input": "input", "output": "output"}

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        messages = [
            Message(
                role="user",
                content=sample[self._column_map["input"]],
                masked=not self.train_on_input,
                eot=True,
            ),
            Message(
                role="assistant",
                content=sample[self._column_map["output"]],
                masked=False,
                eot=True,
            ),
        ]
        if self.new_system_prompt is not None:
            messages = [
                Message(
                    role="system", content=self.new_system_prompt, masked=True, eot=True
                )
            ] + messages
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
        train_on_input (bool): Whether the model is trained on the user prompt or not.
            Default is False.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected
            "chosen" and "rejected" column names to the actual column names in the dataset.
            Default is None, keeping the default column names.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.

    Raises:
        ValueError: If ``column_map`` is provided and ``chosen`` not in ``column_map``, or
            ``rejected`` not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: bool = False,
        column_map: Optional[Dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
    ):
        self.train_on_input = train_on_input
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
            message["masked"] = (message["role"] != "assistant") and (
                not self.train_on_input
            )
            chosen_messages.append(Message.from_dict(message))

        rejected_messages = []
        for message in sample[self._column_map["rejected"]]:
            if message["role"] == "system" and self.new_system_prompt is not None:
                continue
            message["masked"] = (message["role"] != "assistant") and (
                not self.train_on_input
            )
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

        return {"chosen": chosen_messages, "rejected": rejected_messages}


class ShareGPTToMessages(Transform):
    """
    Convert a single chat sample adhering to the ShareGPT json structure to torchtune's :class:`~torchtune.data.Message`
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
        train_on_input (bool): whether the prompt should remain unmasked. Default: False
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns ("conversations")
            to the new column names in the dataset. If None, assume these are identical.
            Default is None.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.

    Raises:
        ValueError: If ``column_map`` is provided and ``conversations`` not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: bool = False,
        column_map: Optional[Dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
    ):
        self.train_on_input = train_on_input
        self.new_system_prompt = new_system_prompt
        if column_map:
            if "conversations" not in column_map:
                raise ValueError(
                    f"Expected a key of 'conversations' in column_map but found {column_map.keys()}."
                )
            self._column_map = column_map
        else:
            self._column_map = {"conversations": "conversations"}

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Return a list of Message objects from the provided sample dict.

        Args:
            sample (Mapping[str, Any]): a single data sample with "messages" field pointing
                to a list of dict messages.

        Returns:
            List[Message]: A list of messages with "role" and "content" fields.
        """
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        messages = []
        if self.new_system_prompt is not None:
            messages.append(
                Message(
                    role="system", content=self.new_system_prompt, masked=True, eot=True
                )
            )
        for message in sample[self._column_map["conversations"]]:
            role = role_map[message["from"]]
            if role == "system" and self.new_system_prompt is not None:
                continue
            content = message["value"]
            masked = (role != "assistant") and (not self.train_on_input)
            messages.append(Message(role=role, content=content, masked=masked))

        return {"messages": messages}


class JSONToMessages(Transform):
    """
    Convert a single chat sample with identical json structure to torchtune's :class:`~torchtune.data.Message`
    structure. This transform simply creates Message dataclasses from the provided jsons.

    A single sample typically consists of a single optional system prompt and one or multiple
    turns of user and assistant messages.

    For example::

        {
            "messages": [
                {
                    "role": <system|user|assistant>,
                    "content": <message>,
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
        train_on_input (bool): whether the prompt should remain unmasked. Default: False
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns ("messages")
            to the new column names in the dataset. If None, assume these are identical.
            Default is None.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.

    Raises:
        ValueError: If ``column_map`` is provided and ``messages`` not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: bool = False,
        column_map: Optional[Dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
    ):
        self.train_on_input = train_on_input
        self.new_system_prompt = new_system_prompt
        if column_map:
            if "messages" not in column_map:
                raise ValueError(
                    f"Expected a key of 'messages' in column_map but found {column_map.keys()}."
                )
            self._column_map = column_map
        else:
            self._column_map = {"messages": "messages"}

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Return a list of Message objects from the provided sample dict.

        Args:
            sample (Mapping[str, Any]): a single data sample with "messages" field pointing
                to a list of dict messages.

        Returns:
            List[Message]: A list of messages with "role" and "content" fields.
        """
        updated_messages = []
        if self.new_system_prompt is not None:
            updated_messages.append(
                Message(
                    role="system", content=self.new_system_prompt, masked=True, eot=True
                )
            )
        for message in sample[self._column_map["messages"]]:
            if message["role"] == "system" and self.new_system_prompt is not None:
                continue
            message["masked"] = (message["role"] != "assistant") and (
                not self.train_on_input
            )
            updated_messages.append(Message.from_dict(message))

        return {"messages": updated_messages}
