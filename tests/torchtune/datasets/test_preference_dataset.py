# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Mapping
from unittest import mock

import pytest
from tests.test_utils import DummyTokenizer
from torchtune.data import Message
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.datasets._preference import PreferenceDataset
from torchtune.modules.transforms import Transform


class ToDummyPreferenceMessages(Transform):
    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        chosen_messages = [
            Message.from_dict(sample["prompt"][0]),
            Message.from_dict(sample["chosen"][0]),
        ]

        rejected_messages = [
            Message.from_dict(sample["prompt"][0]),
            Message.from_dict(sample["rejected"][0]),
        ]

        return {"chosen": chosen_messages, "rejected": rejected_messages}


class TestPreferenceDataset:
    @pytest.fixture
    def dialogue(self):
        return [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "What is 2+2?",
                        "masked": True,
                    },
                ],
                "chosen": [
                    {
                        "role": "assistant",
                        "content": "The answer is 4.",
                        "masked": False,
                    },
                ],
                "rejected": [
                    {
                        "role": "assistant",
                        "content": "The answer is 12.",
                        "masked": False,
                    },
                ],
            },
        ]

    @pytest.fixture
    def expected(self):
        return {
            "prompt": [
                0,
                4,
                2,
                4,
            ],
            "chosen": [
                3,
                6,
                2,
                2,
                -1,
            ],
            "rejected": [
                3,
                6,
                2,
                3,
                -1,
            ],
        }

    @mock.patch("torchtune.datasets._preference.load_dataset")
    def test_get_item(self, mock_load_dataset, dialogue, expected):
        mock_load_dataset.return_value = dialogue
        expected_chosen_tokens = expected["prompt"] + expected["chosen"]
        expected_chosen_labels = [CROSS_ENTROPY_IGNORE_IDX] * len(
            expected["prompt"]
        ) + expected["chosen"]
        expected_rejected_tokens = expected["prompt"] + expected["rejected"]
        expected_rejected_labels = [CROSS_ENTROPY_IGNORE_IDX] * len(
            expected["prompt"]
        ) + expected["rejected"]

        ds = PreferenceDataset(
            source="iam/agoofy/goober",
            message_transform=ToDummyPreferenceMessages(),
            tokenizer=DummyTokenizer(),
        )
        assert len(ds) == 1
        mock_load_dataset.assert_called_once()

        prompt, label = ds[0]["chosen_input_ids"], ds[0]["chosen_labels"]
        assert prompt == expected_chosen_tokens
        assert label == expected_chosen_labels

        prompt, label = ds[0]["rejected_input_ids"], ds[0]["rejected_labels"]
        assert prompt == expected_rejected_tokens
        assert label == expected_rejected_labels
