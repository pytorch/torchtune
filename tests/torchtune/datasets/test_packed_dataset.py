# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tests.test_utils import DummyTokenizer
from torch.utils.data import Dataset

from torchtune.datasets import PackedDataset


class DummyDataset(Dataset):
    def __init__(self, sample_size):
        self.sample_size = sample_size

    def __getitem__(self, index):
        if index >= 1000:
            raise IndexError()
        return [index] * self.sample_size, [index] * self.sample_size


class DummyRealDataset(Dataset):
    def __init__(self):
        self.samples_list = [
            "This is a packing test",
            "A fantastic test. It should pack two samples.",
            "This one will not be fully packed.",
        ]
        self.tokenizer = DummyTokenizer()

    def __getitem__(self, index):
        tokens = self.tokenizer.encode(self.samples_list[index])
        return tokens, tokens


class TestPackedDataset:
    @pytest.mark.parametrize("max_seq_len", [10, 25])
    @pytest.mark.parametrize("sample_size", [2, 5])
    @pytest.mark.parametrize("max_rows", [5, 10])
    @pytest.mark.parametrize("split_samples", [True, False])
    def test_packed_dataset(self, max_seq_len, sample_size, max_rows, split_samples):
        dataset = DummyDataset(sample_size)
        packed = PackedDataset(
            dataset,
            max_seq_len=max_seq_len,
            max_rows=max_rows,
            split_samples=split_samples,
        )
        # Check we get right number of packs
        assert len(packed) == max_rows
        # Check input ids and labels are same length
        assert len(packed[0][0]) == len(packed[0][1])
        # Check that samples are packed correctly - very last individual sample
        # should have index value of the number of times dataset was iterated over
        if split_samples:
            # If we split samples, we'll know how many samples by taking the
            # full length and dividing by sample size
            last_index, remainder = divmod(max_rows * max_seq_len, sample_size)
            # Account for remaining sample that didn't fit in window
            last_index = last_index + 1 if remainder > 0 else last_index
        else:
            # If we don't split samples, we know how many samples by taking
            # how much fits in a single window and multiplying by max rows.
            # We don't account for remainder sample because we'll hit max rows.
            last_index = (max_seq_len // sample_size) * max_rows

        assert packed[-1][0][-1] == last_index - 1

    def test_packed_dataset_real_data(self):
        expected_tokenized_prompts = [
            [0, 4, 2, 1, 7, 4, -1, 0, 1, 9],
            [5, 2, 6, 4, 3, 8, -1, 0, 4, 3],
            [4, 3, 2, 5, 7, -1],
        ]
        packed = PackedDataset(
            DummyRealDataset(),
            max_seq_len=10,
            split_samples=True,
        )

        for i in range(len(packed)):
            prompt, label = packed[i]
            assert prompt == expected_tokenized_prompts[i]
            assert label == expected_tokenized_prompts[i]
