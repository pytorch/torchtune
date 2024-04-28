# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest
from datasets import Dataset
from tests.test_utils import DummyTokenizer
from torchtune.data import Message
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.datasets import ChatDataset


class DummyChatFormat:

    B_SYS, E_SYS = "System:\n", "\n"
    B_INST, E_INST = "User:\n", "\nAssistant:\n"
    B_ASST, E_ASST = "", ""
    system = f"{B_SYS}{{content}}{E_SYS}"
    user = f"{B_INST}{{content}}{E_INST}"
    assistant = f"{B_ASST}{{content}}{E_ASST}"

    @classmethod
    def format(
        cls,
        messages,
    ):
        formats = {"system": cls.system, "user": cls.user, "assistant": cls.assistant}
        formatted_dialogue = []
        for message in messages:
            content = formats.get(message["role"]).format(content=message["content"])
            formatted_dialogue.append(
                Message(
                    role=message["role"], content=content, masked=message["masked"]
                ),
            )
        return formatted_dialogue


def _are_messages_equal(messages_a, messages_b):
    for ma, mb in zip(messages_a, messages_b):
        if ma.role != mb.role:
            return False
        if ma.content != mb.content:
            return False
    return True


class TestChatDataset:
    @pytest.fixture
    def chat_format(self):
        return DummyChatFormat()

    @pytest.fixture
    def dialogue(self):
        return [
            {
                "dialogue": [
                    {
                        "role": "system",
                        "content": "You are an AI assistant.",
                        "masked": True,
                    },
                    {
                        "role": "user",
                        "content": "What is the meaning of life?",
                        "masked": True,
                    },
                    {
                        "role": "assistant",
                        "content": "The meaning of life is 42.",
                        "masked": False,
                    },
                    {"role": "user", "content": "That's ridiculous.", "masked": True},
                    {"role": "assistant", "content": "I agree.", "masked": False},
                ],
            },
        ]

    @mock.patch("torchtune.datasets._chat.load_dataset")
    def test_get_item(self, mock_load_dataset, chat_format, dialogue):
        mock_load_dataset.return_value = Dataset.from_list(dialogue)
        expected_tokenized_prompts = [
            [
                0,
                7,
                3,
                3,
                2,
                2,
                10,
                5,
                4,
                2,
                3,
                7,
                2,
                5,
                10,
                3,
                7,
                2,
                4,
                2,
                3,
                -1,
                0,
                5,
                6,
                11,
                10,
                1,
                6,
                -1,
            ]
        ]
        prompt_lengths = (15, 5)
        expected_labels = [
            [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[0]
            + [3, 7, 2, 4, 2, 3, -1]
            + [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[1]
            + [1, 6, -1]
        ]
        ds = ChatDataset(
            tokenizer=DummyTokenizer(),
            source="iam/agoofy/goober",
            convert_to_messages=lambda x, y: x["dialogue"],
            chat_format=chat_format,
            max_seq_len=100,
            train_on_input=False,
        )
        assert len(ds) == 1
        mock_load_dataset.assert_called_once()

        prompt, label = ds[0]
        assert prompt == expected_tokenized_prompts[0]
        assert label == expected_labels[0]
