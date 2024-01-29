# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune import datasets


class TestDatasetGetter:
    def test_get_dataset(self):
        """
        Test getting a named dataset
        """
        datasets.ALL_DATASETS["test"] = lambda x: x
        dataset = datasets.get_dataset("test", x=1)
        assert dataset == 1

    def test_list_datasets(self):
        """
        Test accuracy of dataset list
        """
        dataset_names = datasets.list_datasets()
        assert "test" in dataset_names
