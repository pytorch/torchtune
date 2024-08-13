# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.test_utils import (
    assert_dialogue_equal,
    CHAT_SAMPLE,
    MESSAGE_SAMPLE,
    MESSAGE_SAMPLE_TRAIN_ON_INPUT,
)
from torchtune.data import get_openai_messages, get_sharegpt_messages


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
        assert_dialogue_equal(converted_messages, MESSAGE_SAMPLE)

    def test_conversion_train_on_input(self):
        converted_messages = get_sharegpt_messages(self.samples, train_on_input=True)
        assert_dialogue_equal(converted_messages, MESSAGE_SAMPLE_TRAIN_ON_INPUT)


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
        assert_dialogue_equal(converted_messages_1, MESSAGE_SAMPLE)

    def test_conversion_messages_key(self):
        converted_messages_2 = get_openai_messages(self.samples_2)
        assert_dialogue_equal(converted_messages_2, MESSAGE_SAMPLE)

    def test_conversion_conversations_key_train_on_input(self):
        converted_messages_1 = get_openai_messages(self.samples_1, train_on_input=True)
        assert_dialogue_equal(converted_messages_1, MESSAGE_SAMPLE_TRAIN_ON_INPUT)

    def test_conversion_messages_key_train_on_input(self):
        converted_messages_2 = get_openai_messages(self.samples_2, train_on_input=True)
        assert_dialogue_equal(converted_messages_2, MESSAGE_SAMPLE_TRAIN_ON_INPUT)
