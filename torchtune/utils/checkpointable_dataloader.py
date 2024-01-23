# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from typing import Any, Dict, Iterator

from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    IterableDataset,
    Sampler,
)


class _FinishedIterWrapper(Iterator):
    def __init__(self, base_iter):
        # Iterator that tracks when it's actually been exhausted
        # (ie __next__ has been called and StopIteration was raised)
        self.base_iter = base_iter
        self.started = False
        self.finished = False

    def __next__(self):
        self.started = True
        try:
            return next(self.base_iter)
        except StopIteration:
            self.finished = True
            raise


class _StatefulSampler:
    def __init__(self, sampler: Sampler):
        self._sampler = sampler
        self.resume_index = 0
        self._iterator = None

    def __len__(self):
        return len(self._sampler)

    def __iter__(self):
        if self._iterator is None:
            it = iter(self._sampler)
            it = itertools.islice(it, self.resume_index, None)
            self._iterator = _FinishedIterWrapper(it)
        elif self._iterator.started:
            # If iter() is called after iteration has started,
            # reset the iterator to the beginning
            self.resume_index = 0
            self._iterator = _FinishedIterWrapper(iter(self._sampler))
        return self._iterator

    def set_state(self, resume_index: int):
        self.resume_index = resume_index
        self._iterator = None


class CheckpointableDataLoader(DataLoader):
    """
    Implements a ``torch.utils.data.DataLoader`` whose state can be
    saved to a checkpoint and restored to resume loading data.

    Currently supports only Map style ``torch.utils.data.Dataset`` and
    ``torch.utils.data.DistributedSampler``.

    Two methods are provided to save and restore the state.

    ``load_state_dict`` restores the state of the dataloader from the given state dict. It should be invoked right after constructing the DataLoader object before any iterator is created from it.

    ``state_dict`` returns the current state of the DataLaoder as a dict.

    Note: As this works only with ``torch.utils.data.DistributedSampler``, the ``set_epoch`` method of that sampler should be invoked before creating an iterator of this class.

    Args:
        dataset (Dataset): ``torch.utils.data.Dataset`` from which to load the data.
        *args: Any arguments to pass to the base DataLoader.
        **kwargs: Any keyword arguments to pass to the base DataLoader.

    Raises:
        ValueError: If the dataset is not a map-style ``torch.util.data.Dataset``.
        ValueError: If the sampler is not a ``torch.utils.data.DistributedSampler``.

    Example:
        >>> sampler = DistributedSampler(...)
        >>> dataloader = CheckpointableDataLoader(dataset, sampler=sampler)
        >>> for epoch in range(0, max_epoch)
        >>>     sampler.set_epoch(epoch)
        >>>     for batch in iter(dataloader):
        >>>         ...
        >>> # Fetch the state of the CheckpointableDataLoader
        >>> state = dataloader.state_dict()
        >>>
        >>> # Restore the state
        >>> dataloader = CheckpointableDataLoader(...)
        >>> dataloader.load_state_dict(state)
        >>> current_epoch = state['epoch'] or 0
        >>> for epoch in range(current_epoch, max_epoch)
        >>>     sampler.set_epoch(epoch)
        >>>     for batch in iter(dataloader):
        >>>         ...
    """  # noqa

    _RESUME_INDEX_KEY = "resume_index"
    _DISTRIBUTED_SAMPLER_SHUFFLE_SEED = "dist_sampler_shuffle_seed"

    def __init__(self, dataset: Dataset, *args, **kwargs):
        self._wrapped_iterator = None
        self._stateful_index_sampler = None
        self._num_yielded = 0
        self._super_iter = None

        if isinstance(dataset, IterableDataset):
            raise ValueError(
                "CheckpointableDataLoader currently supports only map-style dataset. Received an IterableDataset instead."
            )

        super().__init__(dataset, *args, **kwargs)
        if not isinstance(self.sampler, DistributedSampler):
            raise ValueError(
                "CheckpointableDataLoader currently supports only "
                "DistributedSampler. Received a sampler of type "
                f"{type(self.sampler)} instead."
            )

    @property
    def _index_sampler(self):
        if self._stateful_index_sampler is None:
            self._stateful_index_sampler = _StatefulSampler(super()._index_sampler)
        return self._stateful_index_sampler

    def __iter__(self):
        self._super_iter = _FinishedIterWrapper(super().__iter__())
        # Fetch the start index from the sampler's resume index
        self._num_yielded = self._index_sampler.resume_index
        for batch in self._super_iter:
            # Keep track of the new steps iterated through
            self._num_yielded += 1
            yield batch

    def state_dict(self) -> Dict[str, Any]:
        if self._super_iter is None:
            # If no iterator was ever created, return the resumption state
            resume_index = self._index_sampler.resume_index
        elif self._super_iter.finished:
            # If iterator is exhausted, resumption should start from beginning
            resume_index = 0
        else:
            resume_index = self._num_yielded

        return {
            self._RESUME_INDEX_KEY: resume_index,
            self._DISTRIBUTED_SAMPLER_SHUFFLE_SEED: self.sampler.seed,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        resume_index = state_dict[self._RESUME_INDEX_KEY]
        checkpoint_seed = state_dict[self._DISTRIBUTED_SAMPLER_SHUFFLE_SEED]

        # Ensure that the seed of DistributedSampler hasn't changed
        if checkpoint_seed != self.sampler.seed:
            raise AssertionError(
                "On dataloader state load, sampler seed is different - "
                "in sampler '{self.sampler.seed}' != in checkpoint "
                f"{checkpoint_seed}'. Start the run with the seed in the"
                "checkpoint."
            )
        self._index_sampler.set_state(resume_index)
