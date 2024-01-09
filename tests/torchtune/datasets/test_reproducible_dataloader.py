# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torch.utils.data import Dataset, IterableDataset
from torchtune.trainer import ReproducibleDataLoader
from torchtune.utils.env import seed

from tests.test_utils import assert_expected


@pytest.fixture(autouse=True)
def random():
    seed(7)


class InMemoryMapDataset(Dataset):
    def __init__(self, length):
        self._length = length

    def __getitem__(self, index):
        if index >= self._length:
            raise IndexError
        return index

    def __len__(self):
        return self._length


class DummyIterableDataset(IterableDataset):
    def __init__(self, length):
        super().__init__()
        self._length = length

    def __iter__(self):
        for i in range(self._length):
            yield i


class TestReproducibleDataLoader:
    @pytest.mark.parametrize("batch_size,num_workers", [(1, 2), (4, 0), (3, 3)])
    def test_map_dataset_determinism_with_same_seed(self, batch_size, num_workers):
        seed = 12
        map_dataset = InMemoryMapDataset(100)
        results = []

        for run in range(4):
            torch.manual_seed(seed)
            dataloader = ReproducibleDataLoader(
                map_dataset, batch_size=2, shuffle=True, num_workers=4
            )
            for idx, batch in enumerate(dataloader):
                assert_expected(dataloader.sampler.seed, seed)
                if run == 0:
                    results.append(batch)
                else:
                    assert_expected(results[idx], batch)

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_map_dataset_across_epoch_shuffle_data(self, num_workers):
        dataset_size, batch_size = 100, 2
        map_dataset = InMemoryMapDataset(dataset_size)
        results = []
        compare = []

        dataloader = ReproducibleDataLoader(
            map_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: x,
        )
        for run in range(4):
            for idx, batch in enumerate(dataloader):
                if run == 0:
                    results.append(batch)
                else:
                    compare.append(batch)
            if run > 0:
                assert results != compare
                assert len(compare) == len(results) == (dataset_size // batch_size)
                compare = []

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_map_dataset_order_with_no_fixed_seed(self, shuffle):
        dataset_size = 5
        map_dataset = InMemoryMapDataset(dataset_size)
        results = []
        compare = []

        seed = None
        for run in range(4):
            torch.manual_seed(run)
            dataloader = ReproducibleDataLoader(map_dataset, shuffle=shuffle)
            for idx, batch in enumerate(dataloader):
                if run == 0:
                    results.append(batch)
                else:
                    compare.append(batch)
            if run > 0:
                if shuffle:
                    assert results != compare
                else:
                    assert results == compare
                assert len(compare) == len(results) == dataset_size
                compare = []

    def test_iterable_dataset(self):
        # For now make sure initialization with iterable dataset fails
        with pytest.raises(ValueError):
            iterable_dataset = DummyIterableDataset(100)
            dataloader = ReproducibleDataLoader(iterable_dataset)
