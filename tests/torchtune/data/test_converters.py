# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.test_utils import assert_dialogue_equal
from torchtune.data import get_openai_messages, get_sharegpt_messages
from torchtune.data._types import Message

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


class TestShareGPTToLlama2Messages:
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
        converted_messages = get_sharegpt_messages(self.samples)
        assert_dialogue_equal(converted_messages, EXPECTED_MESSAGE)

    def test_conversion_train_on_input(self):
        converted_messages = get_sharegpt_messages(self.samples, train_on_input=True)
        assert_dialogue_equal(converted_messages, EXPECTED_MESSAGE_TRAIN_ON_INPUT)


class TestOpenAIToLlama2Messages:
    samples_1 = {
        "id": "DUMMY",
        "conversations": [
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

    samples_2 = {
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

    def test_conversion_conversations_key(self):
        converted_messages_1 = get_openai_messages(self.samples_1)
        assert_dialogue_equal(converted_messages_1, EXPECTED_MESSAGE)

    def test_conversion_messages_key(self):
        converted_messages_2 = get_openai_messages(self.samples_2)
        assert_dialogue_equal(converted_messages_2, EXPECTED_MESSAGE)

    def test_conversion_conversations_key_train_on_input(self):
        converted_messages_1 = get_openai_messages(self.samples_1, train_on_input=True)
        assert_dialogue_equal(converted_messages_1, EXPECTED_MESSAGE_TRAIN_ON_INPUT)

    def test_conversion_messages_key_train_on_input(self):
        converted_messages_2 = get_openai_messages(self.samples_2, train_on_input=True)
        assert_dialogue_equal(converted_messages_2, EXPECTED_MESSAGE_TRAIN_ON_INPUT)
