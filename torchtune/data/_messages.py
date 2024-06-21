# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, List, Literal, Mapping

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    """
    This dataclass represents individual messages in an instruction or chat dataset.

    Note that the fields ipython and eot are only relevant when tokenizing with tiktoken,
    as they inform handling of special tokens in that case.

    Attributes:
        role (Role): role of the message writer. Can be "system", "user", "assistant".
        content (str): content of the message.
        masked (bool): whether the message is masked in the sample. Default: False
        ipython (bool): whether the message is an ipython call. Default: False
        eot (bool): whether the message corresponds to the end of a turn. Should be true
            except in the case of multiple consecutive assistant messages. Default: True
    """

    role: Role
    content: str
    masked: bool = False
    ipython: bool = False
    eot: bool = True

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


class ShareGptToMessages:
    """
    Convert a chat sample adhering to the ShareGPT json structure to torchtune's :class:`~torchtune.data.Message`
    structure.

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
        sample (Mapping[str, Any]): a single data sample with "conversations" field pointing
            to a list of dict messages.
        train_on_input (bool): whether the prompt should remain unmasked. Default: False

    Returns:
        List[Message]: A list of messages with "role" and "content" fields.
    """

    def __init__(
        self,
        train_on_input: bool = False,
    ):
        self.train_on_input = train_on_input

    def __call__(
        self,
        *,
        conversations: List[Mapping[str, str]],
        **kwargs,
    ) -> Mapping[str, Any]:

        role_map = {"system": "system", "human": "user", "gpt": "assistant"}

        messages = []
        for message in conversations:
            role = role_map[message["from"]]
            content = message["value"]
            masked = (role != "assistant") and (not self.train_on_input)
            messages.append(Message(role=role, content=content, masked=masked))

        return kwargs.update({"messages": messages})


class JsonToMessages:
    """
    Convert a chat sample with identical json structure to torchtune's :class:`~torchtune.data.Message`
    structure. This transform simply creates Message dataclasses from the provided jsons.

    For example::

        {
            # key could be "messages" OR "conversations"
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
        sample (Mapping[str, Any]): a single data sample with "conversations" field pointing
            to a list of dict messages.
        train_on_input (bool): whether the prompt should remain unmasked. Default: False

    Raises:
        ValueError: If the sample does not contain "messages" or "conversations" key.

    Returns:
        List[Message]: A list of messages with "role" and "content" fields.
    """

    def __init__(
        self,
        train_on_input: bool = False,
    ):
        self.train_on_input = train_on_input

    def __call__(
        self,
        *,
        messages: List[Mapping[str, str]],
        **kwargs,
    ) -> Mapping[str, Any]:
        updated_messages = []
        for message in messages:
            message["masked"] = (message["role"] != "assistant") and (
                not self.train_on_input
            )
            updated_messages.append(Message.from_dict(message))

        return kwargs.update({"messages": updated_messages})
