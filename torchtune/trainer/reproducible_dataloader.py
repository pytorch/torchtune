# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Iterable, Optional, Union

import torch
import torch.distributed as dist

from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    IterableDataset,
    Sampler,
)

from torchtune.utils.env import get_world_size_and_rank


class ReproducibleDataLoader(DataLoader):
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
        user. If no seed is provided, a random number is used as the seed.
        (default: ``None``)
    """

    def __init__(  # noqa: DOC101
        self,
        dataset: Dataset,
        shuffle: Optional[bool] = None,
        sampler: Union[Sampler, Iterable, None] = None,
        drop_last: bool = False,
        *args,
        sampler_seed: int,
        **kwargs,
    ):
        """
        Raises:
        ValueError: if dataset is IterableDataset
        """
        if isinstance(dataset, IterableDataset):
            raise ValueError("ReproducibleDataLoader only supports Map style datasets.")

        # Ensure that the seed provided is the same across all ranks
        if dist.is_available() and dist.is_initialized():
            seed_tensor = torch.tensor(sampler_seed)
            dist.barrier()
            output_max = dist.all_reduce(seed_tensor, op=ReduceOp.MAX)
            dist.barrier()
            output_min = dist.all_reduce(seed_tensor, op=ReduceOp.MIN)
            if output_max.item() != sampler_seed or output_min.item() != sampler_seed:
                raise ValueError(
                    f"Seed {sampler_seed} is not the same across all ranks. "
                    f"Max: {output_max.item()}, Min: {output_min.item()}"
                )
        # TODO: Log warning that seed check is not being performed

        self._epoch = 0
        self._is_custom_sampler = sampler is not None
        world_size, rank = get_world_size_and_rank()

        if not self._is_custom_sampler:
            # For map-style dataset, use DistributedSampler that ensures that
            # seed can be provided and shuffling order can be different at
            # epoch intervals
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                seed=sampler_seed,
                drop_last=drop_last,
            )

        super().__init__(
            dataset=dataset,
            shuffle=None,  # shuffle is handled by sampler
            sampler=sampler,
            drop_last=drop_last,
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
