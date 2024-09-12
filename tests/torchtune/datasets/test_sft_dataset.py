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
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms import Transform


class ToDummyMessages(Transform):
    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        dialogue = sample["dialogue"]
        messages = [Message.from_dict(d) for d in dialogue]
        return {"messages": messages}


class DummyTokenizerInvalidModelTransform(DummyTokenizer):
    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        sample = super().__call__(sample)
        del sample["tokens"]
        del sample["images"]
        return sample


class TestSFTDataset:
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
                    {
                        "role": "user",
                        "content": "That's ridiculous.",
                        "masked": True,
                    },
                    {"role": "assistant", "content": "I agree.", "masked": False},
                ],
            },
        ]

    @mock.patch("torchtune.datasets._sft.load_dataset")
    def test_get_item(self, mock_load_dataset, dialogue):
        mock_load_dataset.return_value = dialogue
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
        ds = SFTDataset(
            source="iam/agoofy/goober",
            message_transform=ToDummyMessages(),
            model_transform=DummyTokenizer(),
        )
        assert len(ds) == 1
        mock_load_dataset.assert_called_once()
        prompt, label = ds[0]["tokens"], ds[0]["labels"]
        assert prompt == expected_tokenized_prompts[0]
        assert label == expected_labels[0]

    @pytest.fixture
    def invalid_dialogue(self):
        return [
            {
                "dialogue": [
                    {
                        "role": "user",
                        "content": "What is the meaning of life?",
                        "masked": True,
                    },
                    {
                        "role": "system",
                        "content": "You are an AI assistant.",
                        "masked": True,
                    },
                ],
            },
        ]

    @mock.patch("torchtune.datasets._sft.load_dataset")
    def test_error_for_invalid_messages(self, mock_load_dataset, invalid_dialogue):
        mock_load_dataset.return_value = invalid_dialogue

        ds = SFTDataset(
            source="iam/agoofy/goober",
            message_transform=ToDummyMessages(),
            model_transform=DummyTokenizer(),
        )

        msg = "system messages must come first"
        with pytest.raises(ValueError, match=msg):
            ds[0]

    @mock.patch("torchtune.datasets._sft.load_dataset")
    def test_error_for_invalid_tokenized_dict(self, mock_load_dataset, dialogue):
        mock_load_dataset.return_value = dialogue

        ds = SFTDataset(
            source="iam/agoofy/goober",
            message_transform=ToDummyMessages(),
            model_transform=DummyTokenizerInvalidModelTransform(),
        )

        msg = "model_transform returned the following keys: mask. Must return 'tokens' and 'mask' as keys."
        with pytest.raises(ValueError, match=msg):
            ds[0]
