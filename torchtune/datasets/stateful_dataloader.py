# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from typing import Any, Dict

from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    IterableDataset,
    Sampler,
)


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
    DISTRIBUTED_SAMPLER_SHUFFLE_SEED = "dist_sampler_shuffle_seed"

    def __init__(self, dataset: Dataset, *args, **kwargs):
        self._wrapped_iterator = None
        self._stateful_index_sampler = None
        self._checkpoint_index = 0
        self._num_yielded = 0

        if isinstance(dataset, IterableDataset):
            raise ValueError(
                "StatefulDataLoader currently supports only map-style dataset"
            )

        super().__init__(dataset, *args, **kwargs)
        if not isinstance(self.sampler, DistributedSampler):
            raise ValueError(
                "StatefulDataLoader currently supports only DistributedSampler"
            )

    @property
    def _index_sampler(self):
        if self._stateful_index_sampler is None:
            self._stateful_index_sampler = _StatefulSampler(super()._index_sampler)
        return self._stateful_index_sampler

    def __iter__(self):
        self._num_yielded = self._checkpoint_index
        self._checkpoint_index = 0
        for batch in super().__iter__():
            self._num_yielded += 1
            yield batch

    def state_dict(self) -> Dict[str, Any]:
        resume_index = (
            self._num_yielded if self._checkpoint_index == 0 else self._checkpoint_index
        )
        return {
            self.RESUME_INDEX_KEY: resume_index,
            self.DISTRIBUTED_SAMPLER_SHUFFLE_SEED: self.sampler.seed,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._checkpoint_index = state_dict.get(self.RESUME_INDEX_KEY)
        checkpoint_seed = state_dict.get(self.DISTRIBUTED_SAMPLER_SHUFFLE_SEED)

        # Ensure that the seed of DistributedSampler hasn't changed
        if checkpoint_seed != self.sampler.seed:
            raise AssertionError(
                f"On dataloader state load, sampler seed is different - in sampler '{self.sampler.seed}' != in checkpoint '{checkpoint_seed}'. Start the run with the seed in the checkpoint."  # noqa
            )
        self._index_sampler.set_state(self._checkpoint_index)
