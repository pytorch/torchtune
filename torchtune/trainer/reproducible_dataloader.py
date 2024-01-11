# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Iterable, Optional, Union

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    IterableDataset,
    Sampler,
)
from torch.utils.data.dataloader import _get_distributed_settings


class ReproducibleDataLoader(DataLoader, Stateful):
    """
    A version of :class:`~torch.utils.data.DataLoader` that supports
    reproducing the order of iteration over the dataset and shuffling the
    iteration order across epoch boundaries through the ``seed`` parameter.
    This provides repeatability in dataloading and transforms executed in dataloader workers.

    [Typical usage] If users don't pass in a sampler, then :class:`~torch.utils.
    data.DistributedSampler` is used and its `set_epoch` method is called every time iter is called on the `DataLoader.

    If users provide a custom sampler, then the sampler will be used as is and
    user is responsible for managing that sampler, including setting epoch. No
    reproducibility is guaranteed in this case.

    See :class:`~torch.utils.data.DataLoader` for arguments except ``seed``

    Args:
        seed (int, optional): Seed used to initialize a :class:`~torch.utils.
        data.DistributedSampler` sampler if no custom sampler is provided by the
        user. If no generator is provided, seed is also used to set the
        base_seed for all dataloader workers to ensure transforms are
        repeatable. If no seed is provided, a random number is used as the seed.
        (default: ``None``)
    """

    VERSION_KEY = "state_dict_version"
    EPOCH_COUNT_KEY = "epoch_count"
    SAMPLER_SEED_KEY = "sampler_seed"
    WORLD_SIZE_KEY = "world_size"
    SKIP_NUM_SAMPLES_KEY = "skip_num_samples"
    VERSION_V1 = "v1"

    def __init__(  # noqa: DOC101
        self,
        dataset: Dataset,
        shuffle: Optional[bool] = None,
        sampler: Union[Sampler, Iterable, None] = None,
        drop_last: bool = False,
        generator=None,
        *args,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Raises:
        ValueError: if dataset is IterableDataset
        """
        if isinstance(dataset, IterableDataset):
            raise ValueError("ReproducibleDataLoader only supports Map style datasets.")

        self._epoch = 0
        self._is_custom_sampler = sampler is not None
        # If seed is not set, set it to a random number
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self._seed = seed
        # TODO: Log the seed value for debugging purposes

        # Use the seed as base_seed for all workers to ensure transforms are repeatable
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        self._world_size, rank = _get_distributed_settings()
        if not self._is_custom_sampler:
            # For map-style dataset, use DistributedSampler that ensures that
            # seed can be provided and shuffling order can be different at
            # epoch intervals
            sampler = DistributedSampler(
                dataset,
                num_replicas=self._world_size,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )

        super().__init__(
            dataset=dataset,
            shuffle=None,  # shuffle is handled by sampler
            sampler=sampler,
            drop_last=drop_last,
            generator=generator,
            *args,
            **kwargs,
        )

    def __iter__(self):
        # For every iterator creation, need to set the epoch for the sampler
        # if it is a DistributedSampler to ensure shuffling order is different # across epochs.
        #
        # Note: This is making an assumption that every time an iterator is
        # created, it is start of a new epoch.
        if not self._is_custom_sampler and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(self._epoch)
            self._epoch += 1
        return super().__iter__()

    def state_dict(self) -> Dict[str, Any]:
        sd = {
            self.VERSION_KEY: self.VERSION_V1,
            self.EPOCH_COUNT_KEY: self._epoch,
            self.SAMPLER_SEED_KEY: self._seed,
            self.WORLD_SIZE_KEY: self._world_size,
            self.SKIP_NUM_SAMPLES_KEY: 0,
        }
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]):
        version = state_dict.get(self.VERSION_KEY, None)
        if version != self.VERSION_V1:
            raise ValueError(f"Version key {version} unknown")
        self._epoch = state_dict.get(self.EPOCH_COUNT_KEY)
        seed = state_dict.get(self.SAMPLER_SEED_KEY)
        if seed != self._seed:
            raise ValueError(
                f"Seed {seed} in state_dict does not match seed {self._seed} in loader"
            )

        world_size = state_dict.get(self.WORLD_SIZE_KEY)
        if world_size != self._world_size:
            raise ValueError(
                f"World size {world_size} in state_dict does not match world size {self._world_size} in loader"
            )

        skip_num_samples = state_dict.get(self.SKIP_NUM_SAMPLES_KEY)
