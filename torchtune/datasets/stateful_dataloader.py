# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributed.checkpoint.stateful import Stateful

from torch.utils.data import DataLoader, Dataset, DistributedSampler


class _DataLoaderIteratorWrapper(Iterator):
    def __init__(self, base_iterator: iterator, num_yielded: int = 0):
        self.num_yielded = num_yielded
        self._base_iterator = base_iterator

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._base_iterator)
        self.num_yielded += 1
        return item

    def __len__(self):
        return len(self._base_iterator)

    def __getstate__(self):
        return self._base_iterator.__getstate__()


class StatefulDataLoader(DataLoader, Stateful):
    def __init__(self, dataset: Dataset, sampler: DistributedSampler, *args, **kwargs):
        self._sampler = sampler
        self._wrapped_iterator = None
        super().__init__(dataset, sampler, *args, **kwargs)

    def __iter__(self):
        self._wrapped_iterator = _DataLoaderIteratorWrapper(super().__iter__())
        return self._wrapped_iterator

    def state_dict(self) -> Dict[str, Any]:
        num_yielded = (
            self._wrapped_iterator.num_yielded if self._wrapped_iterator else 0
        )
        sd = {
            self.EPOCH_COUNT_KEY: self._sampler.epoch,
            self.SKIP_NUM_SAMPLES_KEY: num_yielded,
        }
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # TODO: Need a check that world size hasn't changed
        self._sampler.set_epoch(state_dict.get(self.EPOCH_COUNT_KEY))
        skip_num_samples = state_dict.get(self.SKIP_NUM_SAMPLES_KEY)
