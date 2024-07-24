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
from torchtune.datasets import ChatPreferenceDataset


class TestChatPreferenceDataset:
    @pytest.fixture
    def chat_format(self):
        return DummyChatFormat

    @pytest.fixture
    def dialogue(self):
        return [
            Message.from_dict(
                {
                    "role": "user",
                    "content": "Rockin', rockin', and rollin'",
                    "masked": True,
                }
            ),
            Message.from_dict(
                {
                    "role": "assistant",
                    "content": "Down to the beach, I am strolling",
                    "masked": False,
                }
            ),
            Message.from_dict(
                {
                    "role": "user",
                    "content": "But the seagulls poke at my head, not fun!",
                    "masked": True,
                }
            ),
            Message.from_dict(
                {
                    "role": "assistant",
                    "content": "I said, `Seagulls, mm! Stop it now!`",
                    "masked": False,
                }
            ),
        ]

    @pytest.fixture
    def sample(self, dialogue):
        return [
            {
                "chosen": dialogue,
                "rejected": dialogue,
            }
        ]

    @mock.patch("torchtune.datasets._preference.load_dataset")
    def test_get_item(self, mock_load_dataset, chat_format, sample):
        mock_load_dataset.return_value = sample
        expected_tokenized_prompts = [
            [
                0,
                5,
                8,
                8,
                3,
                7,
                10,
                4,
                2,
                3,
                6,
                1,
                2,
                9,
                -1,
                0,
                5,
                3,
                3,
                8,
                4,
                2,
                2,
                5,
                3,
                4,
                10,
                1,
                5,
                10,
                3,
                4,
                2,
                5,
                -1,
            ]
        ]
        prompt_lengths = (7, 12)
        expected_labels = [
            [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[0]
            + [
                4,
                2,
                3,
                6,
                1,
                2,
                9,
                -1,
            ]
            + [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[1]
            + [1, 5, 10, 3, 4, 2, 5, -1]
        ]
        ds = ChatPreferenceDataset(
            tokenizer=DummyTokenizer(),
            source="stop/it/now",
            convert_to_messages=lambda x, y: x["conversations"],
            chat_format=chat_format,
            max_seq_len=100,
            train_on_input=False,
        )
        mock_load_dataset.assert_called_once()

        chosen_prompts, chosen_labels = (
            ds[0]["chosen_input_ids"],
            ds[0]["chosen_labels"],
        )
        rejected_prompts, rejected_labels = (
            ds[0]["rejected_input_ids"],
            ds[0]["rejected_labels"],
        )

        assert chosen_prompts == rejected_prompts == expected_tokenized_prompts[0]
        assert chosen_labels == rejected_labels == expected_labels[0]
