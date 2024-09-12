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

from torchtune.datasets import grammar_dataset


class TestGrammarDataset:
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
                    "input": "Bitcoin is for $7,094 this morning, which CoinDesk says.",
                    "output": "Bitcoin goes for $7,094 this morning, according to CoinDesk.",
                }
            ]
        )

        grammar_ds = grammar_dataset(tokenizer=tokenizer, train_on_input=True)
        input, labels = grammar_ds[0]["tokens"], grammar_ds[0]["labels"]

        assert input == [0, 7, 2, 3, 6, 4, 8, 5, 8, 5, 7, 4, 3, 6, 4, 8, 9, 2, 9, -1]
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
                    "input": "Bitcoin is for $7,094 this morning, which CoinDesk says.",
                    "output": "Bitcoin goes for $7,094 this morning, according to CoinDesk.",
                }
            ]
        )

        grammar_ds = grammar_dataset(tokenizer=tokenizer)

        # Generate the input and labels
        input, labels = grammar_ds[0]["tokens"], grammar_ds[0]["labels"]

        assert input == [0, 7, 2, 3, 6, 4, 8, 5, 8, 5, 7, 4, 3, 6, 4, 8, 9, 2, 9, -1]
        # Check that the input is masked
        assert labels.count(CROSS_ENTROPY_IGNORE_IDX) == 10
