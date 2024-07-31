# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest
from tests.test_utils import DummyChatFormat, DummyTokenizer
from torchtune.data import Message
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.datasets import ChatDataset


class TestChatDataset:
    @pytest.fixture
    def chat_format(self):
        return DummyChatFormat

    @pytest.fixture
    def dialogue(self):
        return [
            {
                "dialogue": [
                    Message.from_dict(
                        {
                            "role": "system",
                            "content": "You are an AI assistant.",
                            "masked": True,
                        }
                    ),
                    Message.from_dict(
                        {
                            "role": "user",
                            "content": "What is the meaning of life?",
                            "masked": True,
                        }
                    ),
                    Message.from_dict(
                        {
                            "role": "assistant",
                            "content": "The meaning of life is 42.",
                            "masked": False,
                        }
                    ),
                    Message.from_dict(
                        {
                            "role": "user",
                            "content": "That's ridiculous.",
                            "masked": True,
                        }
                    ),
                    Message.from_dict(
                        {"role": "assistant", "content": "I agree.", "masked": False}
                    ),
                ],
            },
        ]

    @mock.patch("torchtune.datasets._chat.load_dataset")
    def test_get_item(self, mock_load_dataset, chat_format, dialogue):
        mock_load_dataset.return_value = dialogue
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

        prompt, label = ds[0]["tokens"], ds[0]["labels"]
        assert prompt == expected_tokenized_prompts[0]
        assert label == expected_labels[0]
