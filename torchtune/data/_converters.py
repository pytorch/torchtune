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
    Convert a chat sample adhering to the ShareGPT format to the LLaMA2 format.

    ShareGPT follows:
        {
            "conversations": [
                {
                    "from": <system|human|gpt>,
                    "value": <message>,
                },
                ...
            ]
        }

    LLaMA2 follows:
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
        List[Message]: a list of messages with "role" and "content" fields. See `torchtune.datasets._types.Message`
            for more details.
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
