# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

from torch.utils.data import Dataset, Subset

from torchtune import utils

from torchtune.datasets._packed import PackedDataset

import random

log = utils.get_logger("DEBUG")


def subsample_dataset(dataset, size):
    """
    Subsamples a given dataset to the specified size.

    Args:
        dataset (Dataset): The dataset to subsample.
        size (int): The desired size of the subsample.

    Returns:
        Subset: A new dataset containing the subsampled elements.
    """
    if size > len(dataset):
        raise ValueError("Requested subsample size is larger than the dataset size.")

    # Note: Setting seed for reproducibility.
    random.seed(42)
    indices = random.sample(range(len(dataset)), size)
    return Subset(dataset, indices)


class ConcatDataset(Dataset):
    """
    A dataset class for concatenating multiple sub-datasets into a single dataset. This class enables the
    unified handling of different datasets as if they were a single dataset, simplifying tasks such as
    training models on multiple sources of data simultaneously.

    The class internally manages the aggregation of different datasets and allows transparent indexing across them.
    However, it requires all constituent datasets to be fully loaded into memory, which might not be optimal for
    very large datasets.

    Upon initialization, this class computes the cumulative length of all datasets and maintains an internal mapping
    of indices to the respective datasets. This approach allows the :class:`~torchtune.datasets.ConcatDataset`
    to delegate data retrieval to the appropriate sub-dataset transparently when a particular index is accessed.

    Note:
        Using this class with very large datasets can lead to high memory consumption, as it requires all datasets to
        be loaded into memory. For large-scale scenarios, consider other strategies that might stream data on demand.

    Args:
        datasets (List[Dataset]): A list of datasets to concatenate. Each dataset must be an instance of a class
            derived from :class:`~torch.utils.data.Dataset`.
        portions (List[float]): A list of portions for each dataset. The first dataset is used entirely, and the rest
            are sampled according to the given portions. E.g., if `portions = [0.7, 0.2, 0.1]`, the first dataset is used
            entirely, the second dataset will construct 20% of the total dataset, and the third dataset will construct 10%
            of the total dataset. This means the first dataset will construct 70% of the total dataset. The first portion
            can be anything. The rest of the portions must sum to less than 1.0. if `portions = None`, all datasets are
            used entirely.

    Raises:
        ValueError: if instanse of `PackedDataset` is in `datasets`
        ValueError: if the number of datasets does not match the number of portions
        ValueError: if any portion is not between 0 and 1

    Examples:
        >>> dataset1 = MyCustomDataset(params1)
        >>> dataset2 = MyCustomDataset(params2)
        >>> concat_dataset = ConcatDataset([dataset1, dataset2], [0.5, 0.5])
        >>> print(len(concat_dataset))  # Total length of both datasets
        >>> data_point = concat_dataset[1500]  # Accesses an element from the appropriate dataset


        >>> dataset1 = MyCustomDataset(params1)
        >>> dataset2 = MyCustomDataset(params2)
        >>> dataset3 = MyCustomDataset(params3)
        >>> concat_dataset = ConcatDataset([dataset1, dataset2, dataset3], [0.7, 0.2, 0.1])

    This can also be accomplished by passing in a list of datasets to the YAML config::

        dataset:
          - _component_: torchtune.datasets.instruct_dataset
            source: vicgalle/alpaca-gpt4
            split: train
            train_on_input: True
          - _component_: torchtune.datasets.instruct_dataset
            source: samsum
            column_map: {"output": "summary"}
            split: train
            train_on_input: False


    Or by passing in a list of datasets to the YAML config with portions::

        dataset:
            - _component_: torchtune.datasets.instruct_dataset
                source: vicgalle/alpaca-gpt4
                split: train
                train_on_input: True
                portion: 0.7
            - _component_: torchtune.datasets.instruct_dataset
                source: samsum
                column_map: {"output": "summary"}
                split: train
                train_on_input: False
                portion: 0.3


    This class primarily focuses on providing a unified interface to access elements from multiple datasets,
    enhancing the flexibility in handling diverse data sources for training machine learning models.
    """

    def __init__(self, datasets: List[Dataset], portions: List[float] = None):
        self._datasets: List[Dataset] = datasets
        self._portions: List[float] = portions

        for dataset in self._datasets:
            if isinstance(dataset, PackedDataset):
                raise ValueError(
                    "ConcatDataset can't process instances of PackedDataset."
                )

        if portions is not None:
            if len(datasets) != len(portions):
                raise ValueError(
                    "The number of datasets must match the number of portions."
                )
                # NOTE: Design choice to raise error or use a default portion for the datasets.

            if not sum(portions[1:]) < 1:
                raise ValueError("Sum of portions[1:] must be less than 1.")

            main_dataset_size = len(datasets[0])
            remaining_portion = 1 - sum(portions[1:])
            self._sampled_datasets = [datasets[0]]  # Use all of the first dataset

            for i in range(1, len(datasets)):
                sample_size = int(main_dataset_size * portions[i] / remaining_portion)
                self._sampled_datasets.append(
                    subsample_dataset(datasets[i], sample_size)
                )
            self._datasets = self._sampled_datasets

        self._len: int = sum(len(dataset) for dataset in self._datasets)
        self._indexes: List[Tuple[int, int, int]] = []

        # Calculate distribution of indexes in all datasets
        cumulative_index = 0
        for idx, dataset in enumerate(self._datasets):
            next_cumulative_index = cumulative_index + len(dataset)
            self._indexes.append((cumulative_index, next_cumulative_index, idx))
            cumulative_index = next_cumulative_index

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        for start, stop, dataset_index in self._indexes:
            if start <= index < stop:
                dataset = self._datasets[dataset_index]
                return dataset[index - start]

    def __len__(self) -> int:
        return self._len
