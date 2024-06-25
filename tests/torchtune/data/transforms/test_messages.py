# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.data._types import Message
from torchtune.data.transforms import JsonMessages, ShareGptMessages

# Taken from Open-Orca/SlimOrca-Dedup on Hugging Face:
# https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup
CHAT_SAMPLE = {
    "system": "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",  # noqa: B950
    "user": "Please briefly summarize this news article:\n\nAOL.com Video - Father Lets 8-Year-Old Drive On Icy Road\n\nDescription:Would you let your 8-year-old drive your car? How about on an icy road? Well one father in Russia did just that, and recorded the entire thing. To her credit, the child seemed to be doing a great job. (0:44)\n\nTags: 8-year-old driver , caught on camera , child driver , pix11\n\nSummary:",  # noqa: B950
    "assistant": "A father in Russia allowed his 8-year-old child to drive his car on an icy road and recorded the event. The child appeared to be handling the situation well, showcasing their driving skills despite the challenging conditions.",  # noqa: B950
}

EXPECTED_MESSAGE_TRAIN_ON_INPUT = [
    Message(
        role="system",
        content=CHAT_SAMPLE["system"],
    ),
    Message(
        role="user",
        content=CHAT_SAMPLE["user"],
    ),
    Message(
        role="assistant",
        content=CHAT_SAMPLE["assistant"],
    ),
]

EXPECTED_MESSAGE = [
    Message(role="system", content=CHAT_SAMPLE["system"], masked=True),
    Message(role="user", content=CHAT_SAMPLE["user"], masked=True),
    Message(
        role="assistant",
        content=CHAT_SAMPLE["assistant"],
    ),
]


class TestShareGptMessages:
    samples = {
        "conversations": [
            {
                "from": "system",
                "value": CHAT_SAMPLE["system"],
            },
            {
                "from": "human",
                "value": CHAT_SAMPLE["user"],
            },
            {
                "from": "gpt",
                "value": CHAT_SAMPLE["assistant"],
            },
        ]
    }

    def test_conversion(self):
        transform = ShareGptMessages()
        converted_messages = transform(self.samples)
        for converted, expected in zip(converted_messages, EXPECTED_MESSAGE):
            assert converted == expected

    def test_conversion_train_on_input(self):
        transform = ShareGptMessages(train_on_input=True)
        converted_messages = transform(self.samples)
        for converted, expected in zip(
            converted_messages, EXPECTED_MESSAGE_TRAIN_ON_INPUT
        ):
            assert converted == expected


class TestJsonMessages:
    samples = {
        "id": "DUMMY",
        "messages": [
            {
                "role": "system",
                "content": CHAT_SAMPLE["system"],
            },
            {
                "role": "user",
                "content": CHAT_SAMPLE["user"],
            },
            {
                "role": "assistant",
                "content": CHAT_SAMPLE["assistant"],
            },
        ],
    }

    def test_conversion(self):
        transform = JsonMessages()
        converted_messages_2 = transform(self.samples)
        for converted, expected in zip(converted_messages_2, EXPECTED_MESSAGE):
            assert converted == expected

    def test_conversion_train_on_input(self):
        transform = JsonMessages(train_on_input=True)
        converted_messages_2 = transform(self.samples)
        for converted, expected in zip(
            converted_messages_2, EXPECTED_MESSAGE_TRAIN_ON_INPUT
        ):
            assert converted == expected
