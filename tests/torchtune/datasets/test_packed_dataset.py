# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtune.datasets import PackedDataset

class DummyDataset:
    def __init__(self, sample_size):
        self.sample_size = sample_size

    def __getitem__(self, index):
        return [index] * self.sample_size, [index] * self.sample_size

    def __len__(self):
        return 1000

class TestPackedDataset:
    @pytest.mark.parametrize("max_seq_len", [10, 25])
    @pytest.mark.parametrize("sample_size", [2, 3])
    @pytest.mark.parametrize("max_rows", [5, 10])
    def test_packed_dataset(self, max_seq_len, sample_size, max_rows):
        dataset = DummyDataset(sample_size)
        packed = PackedDataset(
            dataset,
            max_seq_len=max_seq_len,
            max_rows=max_rows,
        )
        # Check we get right number of packs
        assert len(packed) == max_rows
        # Check input ids and labels are same length
        assert len(packed[0][0]) == len(packed[0][1])
        # Check that samples are packed correctly - very last sample
        # should be of length max_seq_len and index value of the number
        # of times dataset was iterated over
        assert packed[-1][0] == [max_rows * max_seq_len // sample_size] * max_seq_len
