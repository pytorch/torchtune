# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torch.utils.data import Dataset
from torchtune.datasets import StatefulDataLoader


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
        dataloader = StatefulDataLoader(dataset)
        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 0

        it = iter(dataloader)

        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 0
        for _ in range(3):
            data = next(it)

        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 3

        dataloader2 = StatefulDataLoader(dataset)
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
        dataloader = StatefulDataLoader(dataset)
        state = dataloader.state_dict()

        it = iter(dataloader)
        data = next(it)

        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 1

        dataloader2 = StatefulDataLoader(dataset)
        dataloader2.load_state_dict(state)
        state = dataloader2.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 1

    def test_across_epoch_state(self):
        dataset = _IdentityMapDataset(8)
        dataloader = StatefulDataLoader(dataset)

        for epoch in range(2):
            if epoch == 0:
                state = dataloader.state_dict()
                assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 0
            for idx, data in enumerate(iter(dataloader)):
                assert data == idx

    def test_save_load_checkpoint_at_end(self):
        dataset = _IdentityMapDataset(2)
        dataloader = StatefulDataLoader(dataset)

        it = iter(dataloader)
        data = next(it)
        data = next(it)

        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 2

        with pytest.raises(StopIteration):
            dataloader2 = StatefulDataLoader(dataset)
            dataloader2.load_state_dict(state)
            it = iter(dataloader2)
            data = next(it)

        it = iter(dataloader)
        data = next(it)
        assert data == 0
        state = dataloader.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 1

        dataloader2 = StatefulDataLoader(dataset)
        dataloader2.load_state_dict(state)
        it = iter(dataloader2)
        data = next(it)
        assert data == 1
        state = dataloader2.state_dict()
        assert state.get(StatefulDataLoader.RESUME_INDEX_KEY) == 2

    @pytest.mark.parametrize("persistent_workers", [False, True])
    def test_multi_worker_state(self, persistent_workers):
        dataset = _IdentityMapDataset(10)
        dataloader = StatefulDataLoader(
            dataset,
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
