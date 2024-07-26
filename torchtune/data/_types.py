# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Literal, Union

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

    Attributes:
        role (Role): role of the message writer. Can be "system", "user", "assistant", or "ipython".
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
            raise RuntimeError(
                f"Media tokens in tool calls are not supported. Both are set in message: {self.text_content}"
            )
        if self.ipython and self.role != "assistant":
            raise RuntimeError(
                f"Only assistant messages can be tool calls. Found role {self.role} in message: {self.text_content}"
            )
