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

    @patch("torchtune.datasets._instruct.load_dataset")
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

        assert len(input) == len(labels)
        assert labels[-1] == tokenizer.eos_id
        assert input[0] == tokenizer.bos_id
        assert CROSS_ENTROPY_IGNORE_IDX not in labels

    @patch("torchtune.datasets._instruct.load_dataset")
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

        # Extract the prompt and tokenize it; we'll need this to test whether we're masking the
        # input correctly
        sample = samsum_ds._data[0]
        prompt = samsum_ds.template.format(sample=sample)
        encoded_prompt = tokenizer.encode(text=prompt, add_bos=True, add_eos=False)

        # Generate the input and labels
        input, labels = samsum_ds[0]["tokens"], samsum_ds[0]["labels"]

        assert len(input) == len(labels)
        assert labels[-1] == tokenizer.eos_id
        assert input[0] == tokenizer.bos_id
        assert labels.count(CROSS_ENTROPY_IGNORE_IDX) == len(encoded_prompt)
