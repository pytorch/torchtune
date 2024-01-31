# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from enum import Enum

from typing import Any, Dict

from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    IterableDataset,
    Sampler,
)

_EpochState = Enum("_EpochState", ("NOT_STARTED", "STARTED", "ENDED"))


class _SkippableSampler:
    """
    Allows skipping a certain number of samples from the beginning of the sampler iterator
    """

    def __init__(self, sampler: Sampler):
        self._sampler = sampler
        self._skip_index = 0
        self._epoch_state = _EpochState.NOT_STARTED

    def __len__(self):
        return len(self._sampler)

    def __iter__(self):
        it = iter(self._sampler)
        if self._epoch_state == _EpochState.NOT_STARTED:
            self._epoch_state = _EpochState.STARTED
            it = itertools.islice(it, self._skip_index, None)

        yield from it

        self._epoch_state = _EpochState.ENDED

    def set_skip_index(self, skip_index: int):
        self._skip_index = skip_index
        self._epoch_state = _EpochState.NOT_STARTED


class CheckpointableDataLoader(DataLoader):
    """
    Implements a :class:`~torch.utils.data.DataLoader` whose state can be
    saved to a checkpoint and restored to resume loading data.

    Currently supports only Map style :class:`~torch.utils.data.Dataset` and
    :class:`~torch.utils.data.distributed.DistributedSampler`.

    Two methods are provided to save and restore the state.

    ``load_state_dict`` restores the state of the dataloader from the given
    state dict. It should be invoked right after constructing the DataLoader
    object before any iterator is created from it.

    ``state_dict`` returns the current state of the DataLaoder as a dict.

    .. note::
        As this works only with :class:`~torch.utils.data.distributed.DistributedSampler`, the ``set_epoch`` method of that sampler should be invoked before creating an iterator of this class.

    .. note::
        CheckpointableDataLoader doesn't save/restore RNG state of the trainer process/dataloader workers. This implies on restore from a checkpoint if the same RNG seed is used, the RNG state will be the same as the beginning of the previous run. If no random transformations of data are performed in the :class:`~torch.utils.data.Dataset` passed to this DataLoader, this should not matter.

    Args:
        dataset (Dataset): :class:`~torch.utils.data.Dataset` from which to load the data.
        *args: Any arguments to pass to the base DataLoader.
        **kwargs: Any keyword arguments to pass to the base DataLoader.

    Raises:
        ValueError: If the dataset is not a map-style :class:`~torch.util.data.Dataset`.
        ValueError: If the sampler is not a :class:`~torch.utils.data.distributed.DistributedSampler`.

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

    _SKIP_INDEX_KEY = "skip_index"
    _DISTRIBUTED_SAMPLER_SHUFFLE_SEED = "dist_sampler_shuffle_seed"

    def __init__(self, dataset: Dataset, *args, **kwargs):
        self._skippable_index_sampler = None
        # This indicates the number of batches that have already been yielded
        # during the current epoch. This is what gets saved as the state in
        # `state_dict()`.
        self._num_yielded = 0
        # This indicates the number of batches that were yielded by the
        # previous iterator that was checkpointed, if any.
        # The only time it is non-zero is right after `load_state_dict()`.
        # After `load_state_dict()`, the next call to __iter__() will start
        # from this value.
        self._last_skip_index = 0
        self._epoch_state: _EpochState = _EpochState.NOT_STARTED

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

    # Override the base dataloader implementation to return a skippable sampler
    @property
    def _index_sampler(self):
        if self._skippable_index_sampler is None:
            self._skippable_index_sampler = _SkippableSampler(super()._index_sampler)
        return self._skippable_index_sampler

    def __iter__(self):
        self._epoch_state = _EpochState.STARTED
        self._num_yielded = self._last_skip_index
        self._last_skip_index = 0
        for batch in super().__iter__():
            # Keep track of the subsequents steps iterated
            self._num_yielded += 1
            yield batch
        self._epoch_state = _EpochState.ENDED

    def state_dict(self) -> Dict[str, Any]:
        skip_index = {
            _EpochState.NOT_STARTED: self._last_skip_index,
            _EpochState.STARTED: self._num_yielded,
            _EpochState.ENDED: 0,
        }[self._epoch_state]

        return {
            self._SKIP_INDEX_KEY: skip_index,
            self._DISTRIBUTED_SAMPLER_SHUFFLE_SEED: self.sampler.seed,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        skip_index = state_dict[self._SKIP_INDEX_KEY]
        checkpoint_seed = state_dict[self._DISTRIBUTED_SAMPLER_SHUFFLE_SEED]

        # Ensure that the seed of DistributedSampler hasn't changed
        if checkpoint_seed != self.sampler.seed:
            raise ValueError(
                "On dataloader state load, sampler seed is different - "
                "in sampler '{self.sampler.seed}' != in checkpoint "
                f"{checkpoint_seed}'. Start the run with the seed in the"
                "checkpoint."
            )
        self._index_sampler.set_skip_index(skip_index)
        self._last_skip_index = skip_index
