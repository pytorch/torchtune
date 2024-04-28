# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping

from torchtune.data._types import Message


def sharegpt_to_llama2_messages(
    sample: Mapping[str, Any], train_on_input: bool = False
) -> List[Message]:
    """
    Convert a chat sample adhering to the ShareGPT format to the Llama2 chat format.

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

    Llama2 follows::

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
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    conversations = sample["conversations"]

    messages = []
    for message in conversations:
        role = role_map[message["from"]]
        content = message["value"]
        masked = (role != "assistant") and (not train_on_input)
        messages.append(Message(role=role, content=content, masked=masked))
    return messages


def openai_to_llama2_messages(
    sample: Mapping[str, Any],
    train_on_input: bool = False,
) -> List[Message]:
    """
    Convert a chat sample adhering to the OpenAI API standard chat format to the Llama2 chat format.

    OpenAI API `standard chat format <https://platform.openai.com/docs/guides/text-generation/chat-completions-api>`_ follows::
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

    Llama2 follows::

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
        message["masked"] = (message["role"] != "assistant") and (not train_on_input)
        messages.append(Message.from_dict(message))
    return messages
