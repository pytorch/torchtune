# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.utils.data import Dataset, DistributedSampler, IterableDataset
from torchtune.utils import StatefulDataLoader


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

        # Creating new iterator should reset state of the dataloader
        for _ in range(2):
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
        old_state = dataloader2.state_dict()
        assert old_state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 0
        dataloader2.load_state_dict(state)
        state = dataloader2.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 1

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

        for epoch in range(2):
            sampler.set_epoch(epoch)
            # Iterate through two epochs
            for idx, data in enumerate(iter(dataloader)):
                assert data == expected_data[epoch][idx]
        # Simulate taking a checkpoint after iterations
        state = dataloader.state_dict()

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
        for new_epoch in range(epoch, epoch + 1):
            sampler.set_epoch(new_epoch)
            for idx, data in enumerate(iter(dataloader2)):
                assert data == expected_data[new_epoch][idx]

    @pytest.mark.parametrize("persistent_workers", [False, True])
    def test_data_with_sampler_shuffle(self, persistent_workers):
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
            3: [2, 4, 3, 5, 1, 0, 6, 7],
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

        # Point check to see if returned data is as expected
        it = iter(dataloader2)
        index = 4
        # Iterate through the rest of the epoch
        for data in it:
            assert data == expected_data[2][index]
            index += 1

        # Start another iteration of new epoch to ensure it iterates through
        # full dataset
        sampler.set_epoch(3)
        actual_data = []
        for data in iter(dataloader2):
            actual_data.append(data)
        assert actual_data == expected_data[3]

    @pytest.mark.parametrize(
        "num_workers, persistent_workers", [(0, None), (0, None), (2, False), (2, True)]
    )
    def test_larger_batch_size(self, num_workers, persistent_workers):
        dataset = _IdentityMapDataset(4)
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)

        multiprocessing_context = "forkserver" if num_workers > 0 else None
        dataloader = StatefulDataLoader(
            dataset,
            batch_size=2,
            shuffle=None,
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context,
            persistent_workers=persistent_workers,
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

        dataloader2 = StatefulDataLoader(
            dataset, batch_size=2, shuffle=None, sampler=sampler
        )
        dataloader2.load_state_dict(state)
        it = iter(dataloader2)
        with pytest.raises(StopIteration):
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

        dataloader2 = StatefulDataLoader(
            dataset, batch_size=1, shuffle=None, sampler=sampler
        )
        dataloader2.load_state_dict(state)
        it = iter(dataloader2)
        with pytest.raises(StopIteration):
            data = next(it)

    def test_dataloader_init_value_errors(self):
        dataset = _IdentityMapDataset(10)
        iterable_dataset = _DummyIterableDataset()

        with pytest.raises(ValueError, match=r".*map-style dataset"):
            # Passing in an iterable dataset
            dataloader = StatefulDataLoader(iterable_dataset)
        with pytest.raises(RuntimeError, match=r".*DistributedSampler"):
            # Not passing Distributed Sampler
            dataloader = StatefulDataLoader(dataset)

    def test_seed_not_same_on_resume(self):
        dataset = _IdentityMapDataset(5)
        sampler = DistributedSampler(
            dataset, seed=5, num_replicas=1, rank=0, shuffle=True
        )
        dataloader = StatefulDataLoader(
            dataset, batch_size=1, shuffle=None, sampler=sampler
        )
        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.DISTRIBUTED_SAMPLER_SHUFFLE_SEED) == 5

        # Modify the state and ensure assertion error is thrown on load
        state[StatefulDataLoader.DISTRIBUTED_SAMPLER_SHUFFLE_SEED] = 10
        with pytest.raises(AssertionError):
            dataloader.load_state_dict(state)

    def test_state_contains_expected_keys(self):
        dataset = _IdentityMapDataset(5)
        sampler = DistributedSampler(
            dataset, seed=5, num_replicas=1, rank=0, shuffle=True
        )
        dataloader = StatefulDataLoader(
            dataset, batch_size=1, shuffle=None, sampler=sampler
        )
        # Perform one batch fetch
        it = iter(dataloader)
        next(it)

        # Check that the state contains the expected keys
        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.DISTRIBUTED_SAMPLER_SHUFFLE_SEED) == 5
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 1
        assert len(state) == 2
