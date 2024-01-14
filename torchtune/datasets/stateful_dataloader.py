# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from typing import Any, Dict, Iterator

from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler


class _DataLoaderIteratorWrapper(Iterator):
    def __init__(self, base_iterator: Iterator, resume_index: int):
        self.resume_index = resume_index
        self.base_iterator = base_iterator

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.base_iterator)
        # Increment the resume index as data is consumed
        self.resume_index += 1
        return item

    def __len__(self):
        return len(self.base_iterator)

    def __getstate__(self):
        return self.base_iterator.__getstate__()


class _StatefulSampler:
    def __init__(self, sampler: Sampler):
        self._sampler = sampler
        self.resume_index = 0
        self._iterator = None

    def __len__(self):
        return len(self._sampler)

    def __iter__(self):
        self._iterator = iter(self._sampler)
        # Fast-forward sampler to the resume index if necessary
        if self.resume_index > 0:
            self._iterator = itertools.islice(self._iterator, self.resume_index, None)
            self.resume_index = 0
        return self._iterator

    def set_state(self, resume_index: int):
        self.resume_index = resume_index
        self._iterator = None


class StatefulDataLoader(DataLoader):
    RESUME_INDEX_KEY = "resume_index"

    def __init__(self, dataset: Dataset, *args, **kwargs):
        self._wrapped_iterator = None
        self._stateful_index_sampler = None
        self._resume_index = 0

        if isinstance(dataset, IterableDataset):
            raise ValueError(
                "StatefulDataLoader currently supports only map-style dataset"
            )
        super().__init__(dataset, *args, **kwargs)

    @property
    def _index_sampler(self):
        if self._stateful_index_sampler is None:
            self._stateful_index_sampler = _StatefulSampler(super()._index_sampler)
        return self._stateful_index_sampler

    def __iter__(self):
        # Wrap the base iterator to keep track of the resume index
        self._wrapped_iterator = _DataLoaderIteratorWrapper(
            super().__iter__(), self._resume_index
        )
        self._resume_index = 0

        return self._wrapped_iterator

    def state_dict(self) -> Dict[str, Any]:
        resume_index = (
            self._wrapped_iterator.resume_index
            if self._wrapped_iterator
            else self._resume_index
        )
        sd = {
            self.RESUME_INDEX_KEY: resume_index,
        }
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # TODO: Need a check that world size hasn't changed
        self._resume_index = state_dict.get(self.RESUME_INDEX_KEY)
        self._index_sampler.set_state(self._resume_index)
