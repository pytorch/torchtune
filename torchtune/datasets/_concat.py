# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from torchtune.datasets._concat import ConcatDataset
from torchtune.datasets._packed import PackedDataset


class DummyDataset(TorchDataset):
    def __init__(self, sample_size):
        self.sample_size = sample_size

    def __getitem__(self, index):
        if index >= 1000:
            raise IndexError()
        return {
            "tokens": [index] * self.sample_size,
            "labels": [index] * self.sample_size,
        }

    def __len__(self):
        return 1000


class TestConcatDataset:
    @pytest.fixture
    def datasets(self):
        ds1 = Dataset.from_list([{"data": f"ds1_{i}"} for i in range(4)])
        ds2 = Dataset.from_list([{"data": f"ds2_{i}"} for i in range(8)])
        ds3 = Dataset.from_list([{"data": f"ds3_{i}"} for i in range(15)])
        ds4 = Dataset.from_list([{"data": f"ds4_{i}"} for i in range(16)])
        ds5 = Dataset.from_list([{"data": f"ds5_{i}"} for i in range(23)])
        ds6 = Dataset.from_list([{"data": f"ds6_{i}"} for i in range(42)])
        return [ds1, ds2, ds3, ds4, ds5, ds6]

    @pytest.fixture
    def torch_datasets(self):
        ds1 = DummyDataset(4)
        ds2 = DummyDataset(8)
        ds3 = DummyDataset(15)
        ds4 = DummyDataset(16)
        ds5 = DummyDataset(23)
        ds6 = DummyDataset(42)
        return [ds1, ds2, ds3, ds4, ds5, ds6]

    def test_length(self, datasets):
        """Test the correct computation of total length"""
        multi_dataset = ConcatDataset(datasets)

        # sum of individual datasets lengths
        expected_length = 4 + 8 + 15 + 16 + 23 + 42  # 108
        assert len(multi_dataset) == expected_length

    def test_getitem(self, datasets):
        """Test item retrieval across dataset boundaries"""
        multi_dataset = ConcatDataset(datasets)

        # Testing indices across different datasets
        assert multi_dataset[-1] is None  # Index out of range
        assert multi_dataset[0] == {"data": "ds1_0"}
        assert multi_dataset[3] == {"data": "ds1_3"}
        assert multi_dataset[4] == {"data": "ds2_0"}
        assert multi_dataset[10] == {"data": "ds2_6"}
        assert multi_dataset[20] == {"data": "ds3_8"}
        assert multi_dataset[35] == {"data": "ds4_8"}
        assert multi_dataset[50] == {"data": "ds5_7"}
        assert multi_dataset[70] == {"data": "ds6_4"}
        assert multi_dataset[90] == {"data": "ds6_24"}
        assert multi_dataset[108] is None  # Index out of range

    def test_invalid_index_type(self, datasets):
        """Test handling of invalid index types"""
        multi_dataset = ConcatDataset(datasets)

        with pytest.raises(TypeError):
            multi_dataset["invalid_type"]  # Non-integer index

    def test_packed_dataset(self, torch_datasets):
        torch_datasets[0] = PackedDataset(
            torch_datasets[0],
            max_seq_len=25,
            max_packs=5,
            split_across_pack=True,
        )

        with pytest.raises(ValueError):
            concated_dataset = ConcatDataset(torch_datasets)
