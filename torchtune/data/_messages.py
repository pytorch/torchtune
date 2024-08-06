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
        eot (bool): whether the message corresponds to the end of a turn. Should be true
            except in the case of multiple consecutive assistant messages (i.e., tool calls
            by assistant). Default: True
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
    Message transform class that converts a sample with "input" and "output" fields,
    (or equivalent fields specified in column_map) to user and assistant messages,
    respectively. This is useful for datasets that have two columns, one containing
    the user prompt and the other containing the model response.

    Args:
        train_on_input (bool): Whether the model is trained on the user prompt or not.
            Default is False.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "input"
            and "output" column names to the actual column names in the dataset. Default is None,
            keeping the default "input" and "output" column names.
    """

    def __init__(
        self, train_on_input: bool = False, column_map: Optional[Dict[str, str]] = None
    ):
        self.train_on_input = train_on_input
        self.column_map = column_map

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        column_map = self.column_map or {}
        key_input = column_map.get("input", "input")
        key_output = column_map.get("output", "output")
        messages = [
            Message(
                role="user",
                content=sample[key_input],
                masked=not self.train_on_input,
                eot=False,
            ),
            Message(
                role="assistant",
                content=sample[key_output],
                masked=False,
                eot=True,
            ),
        ]
        return {"messages": messages}


class ChosenRejectedToMessages(Transform):
    def __init__(
        self, train_on_input: bool = False, column_map: Optional[Dict[str, str]] = None
    ):
        self.train_on_input = train_on_input
        self.column_map = column_map

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        column_map = self._column_map or {}
        key_prompt = column_map.get("prompt", "prompt")
        key_chosen = column_map.get("chosen", "chosen")
        key_rejected = column_map.get("rejected", "rejected")

        chosen_messages = [
            Message(role="user", content=sample[key_prompt], masked=True),
            Message(role="assistant", content=sample[key_chosen]),
        ]

        rejected_messages = [
            Message(role="user", content=sample[key_prompt], masked=True),
            Message(role="assistant", content=sample[key_rejected]),
        ]

        return {"chosen": chosen_messages, "rejected": rejected_messages}
