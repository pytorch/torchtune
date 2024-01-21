# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.utils.data import Dataset, DistributedSampler, IterableDataset
from torchtune.datasets import StatefulDataLoader


class _DummyIterableDataset(IterableDataset):
    def __iter__(self):
        yield 1


class _IdentityMapDataset(Dataset):
    def __init__(self, length):
        self._length = length

    def __getitem__(self, index):
        if index >= self._length:
            raise IndexError
        return index

    def __len__(self):
        return self._length


class TestStatefulDataLoader:
    def test_save_checkpoint_state(self):
        dataset = _IdentityMapDataset(10)
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
        dataloader = StatefulDataLoader(
            dataset, batch_size=1, shuffle=None, sampler=sampler
        )
        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 0

        it = iter(dataloader)

        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 0
        for _ in range(3):
            data = next(it)

        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 3

        dataloader2 = StatefulDataLoader(
            dataset, batch_size=1, shuffle=None, sampler=sampler
        )
        dataloader2.load_state_dict(state)
        it = iter(dataloader2)
        data = next(it)

        assert data == 3
        state = dataloader2.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 4

        it = iter(dataloader2)
        data = next(it)

        assert data == 0
        state = dataloader2.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 1

    def test_set_and_load_checkpoint_right_away(self):
        dataset = _IdentityMapDataset(10)
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
        dataloader = StatefulDataLoader(
            dataset, batch_size=1, shuffle=None, sampler=sampler
        )
        state = dataloader.state_dict()

        it = iter(dataloader)
        data = next(it)

        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 1

        dataloader2 = StatefulDataLoader(
            dataset, batch_size=1, shuffle=None, sampler=sampler
        )
        dataloader2.load_state_dict(state)
        state = dataloader2.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 1

    def test_across_epoch_state(self):
        dataset = _IdentityMapDataset(8)
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
        dataloader = StatefulDataLoader(
            dataset, batch_size=1, shuffle=None, sampler=sampler
        )

        for epoch in range(2):
            if epoch == 0:
                state = dataloader.state_dict()
                assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 0
            for idx, data in enumerate(iter(dataloader)):
                assert data == idx

    @pytest.mark.parametrize("persistent_workers", [False, True])
    def test_multiworker_dataloader_epoch_end(self, persistent_workers):
        dataset = _IdentityMapDataset(8)
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
        dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            shuffle=None,
            sampler=sampler,
            num_workers=2,
            persistent_workers=persistent_workers,
            multiprocessing_context="forkserver",
            prefetch_factor=2,
        )

        expected_data = {
            0: [4, 0, 7, 3, 2, 5, 1, 6],
            1: [5, 4, 2, 6, 7, 3, 1, 0],
            2: [0, 4, 7, 2, 6, 5, 1, 3],
        }

        state = None
        current_epoch = None
        for epoch in range(2):
            sampler.set_epoch(epoch)
            current_epoch = epoch
            for idx, data in enumerate(iter(dataloader)):
                assert data == expected_data[epoch][idx]
                state = dataloader.state_dict()
        # Simulate taking a checkpoint

        # New run starts with previously saved state
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
        dataloader2 = StatefulDataLoader(
            dataset,
            batch_size=1,
            shuffle=None,
            sampler=sampler,
            num_workers=2,
            persistent_workers=persistent_workers,
            multiprocessing_context="forkserver",
            prefetch_factor=2,
        )
        dataloader2.load_state_dict(state)
        for epoch in range(current_epoch, current_epoch + 1):
            sampler.set_epoch(epoch)
            for idx, data in enumerate(iter(dataloader2)):
                assert data == expected_data[epoch][idx]

    @pytest.mark.parametrize("persistent_workers", [False, True])
    def test_sample_data(self, persistent_workers):
        dataset = _IdentityMapDataset(8)
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
        dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            shuffle=None,
            sampler=sampler,
            num_workers=2,
            persistent_workers=persistent_workers,
            multiprocessing_context="forkserver",
            prefetch_factor=2,
        )

        expected_data = {
            0: [4, 0, 7, 3, 2, 5, 1, 6],
            1: [5, 4, 2, 6, 7, 3, 1, 0],
            2: [0, 4, 7, 2, 6, 5, 1, 3],
        }

        state = None
        for epoch in range(3):
            sampler.set_epoch(epoch)
            for idx, data in enumerate(iter(dataloader)):
                assert data == expected_data[epoch][idx]
                if epoch == 2 and idx == 3:
                    state = dataloader.state_dict()
                    break

        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
        dataloader2 = StatefulDataLoader(
            dataset,
            batch_size=1,
            shuffle=None,
            sampler=sampler,
            num_workers=2,
            persistent_workers=persistent_workers,
            multiprocessing_context="forkserver",
            prefetch_factor=2,
        )
        dataloader2.load_state_dict(state)
        sampler.set_epoch(2)

        it = iter(dataloader2)
        data = next(it)
        assert data == expected_data[2][4]

    def test_larger_batch_size(self):
        dataset = _IdentityMapDataset(4)
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
        dataloader = StatefulDataLoader(
            dataset, batch_size=2, shuffle=None, sampler=sampler
        )

        it = iter(dataloader)
        data = next(it)
        assert torch.equal(data, torch.tensor([0, 1]))
        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 1
        data = next(it)
        assert torch.equal(data, torch.tensor([2, 3]))
        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 2

        with pytest.raises(StopIteration):
            dataloader2 = StatefulDataLoader(
                dataset, batch_size=2, shuffle=None, sampler=sampler
            )
            dataloader2.load_state_dict(state)
            it = iter(dataloader2)
            data = next(it)

    def test_save_load_checkpoint_at_end(self):
        dataset = _IdentityMapDataset(2)
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
        dataloader = StatefulDataLoader(
            dataset, batch_size=1, shuffle=None, sampler=sampler
        )

        it = iter(dataloader)
        data = next(it)
        data = next(it)

        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 2

        with pytest.raises(StopIteration):
            dataloader2 = StatefulDataLoader(
                dataset, batch_size=1, shuffle=None, sampler=sampler
            )
            dataloader2.load_state_dict(state)
            it = iter(dataloader2)
            data = next(it)

        it = iter(dataloader)
        data = next(it)
        assert data == 0
        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 1

        dataloader2 = StatefulDataLoader(
            dataset, batch_size=1, shuffle=None, sampler=sampler
        )
        dataloader2.load_state_dict(state)
        it = iter(dataloader2)
        data = next(it)
        assert data == 1
        state = dataloader2.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 2

    @pytest.mark.parametrize("persistent_workers", [False, True])
    def test_multi_worker_state(self, persistent_workers):
        dataset = _IdentityMapDataset(10)
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
        dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            shuffle=None,
            sampler=sampler,
            num_workers=2,
            multiprocessing_context="forkserver",
            persistent_workers=persistent_workers,
        )

        for epoch in range(2):
            if epoch == 0:
                state = dataloader.state_dict()
                assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 0
            for idx, data in enumerate(iter(dataloader)):
                assert data == idx
                state = dataloader.state_dict()
                assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == idx + 1

    def test_value_errors(self):
        dataset = _IdentityMapDataset(10)
        iterable_dataset = _DummyIterableDataset()

        with pytest.raises(ValueError):
            # Passing in an iterable dataset
            dataloader = StatefulDataLoader(iterable_dataset)
        with pytest.raises(ValueError):
            # Not passing Distributed Sampler
            dataloader = StatefulDataLoader(dataset)
