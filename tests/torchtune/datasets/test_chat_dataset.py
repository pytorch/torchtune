# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tests.common import ASSETS
from tests.test_utils import DummyChatFormat, DummyTokenizer
from torchtune.data import get_sharegpt_messages
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.datasets import chat_dataset, ChatDataset


class TestChatDataset:
    @pytest.fixture
    def chat_format(self):
        return DummyChatFormat

    def test_get_item(self, chat_format):
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
            source="json",
            convert_to_messages=get_sharegpt_messages,
            chat_format=chat_format,
            max_seq_len=100,
            train_on_input=False,
            data_files=str(ASSETS / "chat_tiny.json"),
            split="train",
        )
        assert len(ds) == 1
        prompt, label = ds[0]["tokens"], ds[0]["labels"]
        assert prompt == expected_tokenized_prompts[0]
        assert label == expected_labels[0]

        expected_tokenized_prompts = [
            [
                0,
                3,
                3,
                2,
                2,
                10,
                4,
                2,
                3,
                7,
                2,
                5,
                3,
                7,
                2,
                4,
                2,
                3,
                -1,
                0,
                6,
                11,
                1,
                6,
                -1,
            ]
        ]
        prompt_lengths = (12, 3)
        expected_labels = [
            [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[0]
            + [3, 7, 2, 4, 2, 3, -1]
            + [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[1]
            + [1, 6, -1]
        ]

        ds = chat_dataset(
            tokenizer=DummyTokenizer(),
            source="json",
            data_files=str(ASSETS / "chat_tiny.json"),
            conversation_column="conversations",
            conversation_style="sharegpt",
            train_on_input=False,
            packed=False,
            split="train",
        )

        assert len(ds) == 1
        prompt, label = ds[0]["tokens"], ds[0]["labels"]
        assert prompt == expected_tokenized_prompts[0]
        assert label == expected_labels[0]
