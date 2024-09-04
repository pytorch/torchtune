# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
from datasets import Dataset

from tests.test_utils import DummyTokenizer
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX

from torchtune.datasets import samsum_dataset


class TestSamsumDataset:
    @pytest.fixture
    def tokenizer(self):
        return DummyTokenizer()

    @patch("torchtune.datasets._sft.load_dataset")
    def test_label_no_masking(self, load_dataset, tokenizer):
        """
        Test whether the input and the labels are correctly created when the input is not masked.
        """

        # mock the call to HF datasets
        load_dataset.return_value = Dataset.from_list(
            [
                {
                    "id": "13818513",
                    "dialogue": "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-)",
                    "summary": "Amanda baked cookies and will bring Jerry some tomorrow.",
                },
            ]
        )

        samsum_ds = samsum_dataset(tokenizer=tokenizer, train_on_input=True)
        input, labels = samsum_ds[0]["tokens"], samsum_ds[0]["labels"]

        assert input == [
            0,
            7,
            1,
            5,
            8,
            2,
            3,
            4,
            5,
            6,
            5,
            7,
            4,
            5,
            3,
            8,
            3,
            6,
            5,
            7,
            3,
            4,
            5,
            5,
            4,
            9,
            -1,
        ]
        assert labels == input

    @patch("torchtune.datasets._sft.load_dataset")
    def test_label_masking(self, load_dataset, tokenizer):
        """
        Test whether the input and the labels are correctly created when the input is masked.
        """

        # mock the call to HF datasets
        load_dataset.return_value = Dataset.from_list(
            [
                {
                    "id": "13818513",
                    "dialogue": "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-)",
                    "summary": "Amanda baked cookies and will bring Jerry some tomorrow.",
                },
            ]
        )

        samsum_ds = samsum_dataset(tokenizer=tokenizer)

        # Generate the input and labels
        input, labels = samsum_ds[0]["tokens"], samsum_ds[0]["labels"]

        assert input == [
            0,
            7,
            1,
            5,
            8,
            2,
            3,
            4,
            5,
            6,
            5,
            7,
            4,
            5,
            3,
            8,
            3,
            6,
            5,
            7,
            3,
            4,
            5,
            5,
            4,
            9,
            -1,
        ]
        assert labels.count(CROSS_ENTROPY_IGNORE_IDX) == 17
