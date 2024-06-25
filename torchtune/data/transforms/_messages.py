# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

from torchtune.data._types import Message


class ShareGptMessages:
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
        sample: Dict[str, Any],
    ) -> List[Message]:

        role_map = {"system": "system", "human": "user", "gpt": "assistant"}

        messages = []
        for message in sample["conversations"]:
            role = role_map[message["from"]]
            content = message["value"]
            masked = (role != "assistant") and (not self.train_on_input)
            messages.append(Message(role=role, content=content, masked=masked))

        return messages


class JsonMessages:
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
        sample: Dict[str, Any],
    ) -> List[Message]:
        if "messages" in sample:
            messages_key = "messages"
        elif "conversations" in sample:
            messages_key = "conversations"
        else:
            raise ValueError(
                f"Sample does not contain 'messages' or 'conversations' key. Existing keys: {sample.keys()}"
            )
        conversations = sample[messages_key]
        messages = []
        for message in conversations:
            message["masked"] = (message["role"] != "assistant") and (
                not self.train_on_input
            )
            messages.append(Message.from_dict(message))

        return messages


class ColumnMessages:
    def __init__(
        self,
        input_column: str,
        output_column: str,
    ):
        self.input_column = input_column
        self.output_column = output_column

    def __call__(
        self,
        sample: Dict[str, Any],
    ) -> List[Message]:
        """
        Generate prompt from sentence that needs grammar correction.

        Args:
            sample (List[Message]): a single conversation, structured as a list
                of `Message` objects

        Returns:
            The formatted list of messages
        """
        messages = []
        messages.append(
            Message(
                role="user",
                content=sample[self.input_column],
            ),
        )
        messages.append(
            Message(role="assistant", content=sample[self.output_column]),
        )
        return messages


class PreferenceMessages:
    def __call__(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, List[Message]]:
        chosen_messages = [
            Message(role="user", content=sample["prompt"], masked=True),
            Message(role="assistant", content=sample["chosen"]),
        ]

        rejected_messages = [
            Message(role="user", content=sample["prompt"], masked=True),
            Message(role="assistant", content=sample["rejected"]),
        ]
        return {"chosen": chosen_messages, "rejected": rejected_messages}
