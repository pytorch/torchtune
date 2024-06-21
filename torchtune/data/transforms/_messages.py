# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping

from torchtune.data._prompt_templates import AlpacaInstructTemplate

from torchtune.data._types import Message


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
        key: str = "conversations",
        train_on_input: bool = False,
    ):
        self.key = key
        self.train_on_input = train_on_input

    def __call__(
        self,
        sample: Mapping[str, Any],
    ) -> Mapping[str, Any]:

        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        conversations = sample[self.key]

        messages = []
        for message in conversations:
            role = role_map[message["from"]]
            content = message["value"]
            masked = (role != "assistant") and (not train_on_input)
            messages.append(Message(role=role, content=content, masked=masked))

        processed_sample = {k: v for k, v in sample.items() if k != self.key}
        processed_sample["text"] = messages
        return processed_sample


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
        key: str = "messages",
        train_on_input: bool = False,
    ):
        self.key = key
        self.train_on_input = train_on_input

    def __call__(
        self,
        sample: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        conversations = sample[self.key]

        messages = []
        for message in conversations:
            message["masked"] = (message["role"] != "assistant") and (
                not train_on_input
            )
            messages.append(Message.from_dict(message))

        processed_sample = {k: v for k, v in sample.items() if k != self.key}
        processed_sample["text"] = messages
        return processed_sample
