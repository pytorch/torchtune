# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Iterable, List, Optional, Tuple, Union

import torch

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    IterableDataset,
    Sampler,
)
from torch.utils.data.dataloader import _get_distributed_settings

# TokenPair is a pair (tuple) of two lists: tokenized text inputs and labels.
TokenPair = Tuple[List[int], List[int]]

_DEFAULT_INPUT_PADDING_IDX: int = 0
_DEFAULT_LABEL_PADDING_IDX: int = -100


class ReproducibleDataLoader(DataLoader):
    """A version of :class:`~torch.utils.data.DataLoader` that supports
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
        dataset (Dataset): dataset from which to load the data.
        shuffle (Optional[bool]): set to True to have the data reshuffled at every epoch, None for sampler override
        sampler (Optional[Union[Sampler, Iterable]]): uses DistributedSampler if no custom sampler is provided
        drop_last (bool): set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
        generator (Optional[torch.Generator]): RNG, if no generator is provided, seed is also used to set the
            base_seed for all dataloader workers to ensure transforms are repeatable
        *args: additional positional arguments from the base DataLoader
        seed (Optional[int]): Seed used to initialize a :class:`~torch.utils.
        **kwargs: additional keyword arguments from the base DataLoader

    Raises:
        ValueError: If the dataset is not a map-style dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: Optional[bool] = None,
        sampler: Optional[Union[Sampler, Iterable]] = None,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
        *args,
        seed: Optional[int] = None,
        **kwargs,
    ):
        if isinstance(dataset, IterableDataset):
            raise ValueError("ReproducibleDataLoader only supports Map style datasets.")

        self._epoch = 0
        self._is_custom_sampler = sampler is not None
        # If seed is not set, set it to a random number
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        # TODO: Log the seed value for debugging purposes

        # Use the seed as base_seed for all workers to ensure transforms are repeatable
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        world_size, rank = _get_distributed_settings()
        if not self._is_custom_sampler:
            # For map-style dataset, use DistributedSampler that ensures that
            # seed can be provided and shuffling order can be different at
            # epoch intervals
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
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


def batch_pad_to_longest_seq(
    batch: List[TokenPair],
    input_padding_idx: int = _DEFAULT_INPUT_PADDING_IDX,
    label_padding_idx: int = _DEFAULT_LABEL_PADDING_IDX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors.

    Args:
        batch (List[TokenPair]): A list of tuples containing input, label pairs.
        input_padding_idx (int): Padding index for input ids. Defaults to 0.
        label_padding_idx (int): Padding index for labels. Defaults to -100.
    Returns:
        Collated input and label tensors.

    Example:
        token_pairs = [
            ([1, 2, 3], [4, 5, 6]),
            ([7,], [10,],),
        ]
        inputs, labels = batch_pad_to_longest_seq(
            batch=token_pairs,
            input_padding_idx=input_padding_idx,
            label_padding_idx=label_padding_idx,
        )
        >>> inputs
            tensor([[1, 2, 3], [7, 0, 0]])
        >>> labels
            tensor([[4,5,6], [10,-100,-100]])
    """
    input_ids = pad_sequence(
        [torch.tensor(x[0]) for x in batch],
        batch_first=True,
        padding_value=input_padding_idx,
    )
    labels = pad_sequence(
        [torch.tensor(x[1]) for x in batch],
        batch_first=True,
        padding_value=label_padding_idx,
    )

    input_ids_seq_len = input_ids.shape[-1]
    labels_seq_len = labels.shape[-1]

    # Hack to pad correctly and not use max_seq_len, which is costly
    if input_ids_seq_len > labels_seq_len:
        labels = F.pad(
            labels, (0, input_ids_seq_len - labels_seq_len), value=label_padding_idx
        )
    elif labels_seq_len > input_ids_seq_len:
        input_ids = F.pad(
            input_ids,
            (0, labels_seq_len - input_ids_seq_len),
            value=input_padding_idx,
        )
    return input_ids, labels
