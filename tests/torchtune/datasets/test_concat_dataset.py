# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from datasets import Dataset
from torchtune.datasets._concat import ConcatDataset


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
