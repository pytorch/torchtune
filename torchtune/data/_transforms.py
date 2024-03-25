# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.data._types import Dialogue, Message, Sample


def sharegpt_to_llama2_dialogue(sample: Sample) -> Dialogue:
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
        sample (Sample): a single data sample with "conversations" field pointing
            to a list of dict messages.

    Returns:
        Dialogue: a list of messages with "role" and "content" fields. See `torchtune.datasets._types.Message`
            and `torchtune.datasets._types.Dialogue` for more details.
    """
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    conversations = sample["conversations"]

    dialogue = []
    for message in conversations:
        role = role_map[message["from"]]
        content = message["value"]
        dialogue.append(Message(role=role, content=content))

    return dialogue
