# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset

from torchtune import utils

from torchtune.datasets._packed import PackedDataset

log = utils.get_logger("DEBUG")


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
        datasets (list[Dataset]): A list of datasets to concatenate. Each dataset must be an instance of a class
            derived from :class:`~torch.utils.data.Dataset`.

    Raises:
        ValueError: if a mix of `PackedDataset` and non-`PackedDataset` datasets is provided

    Examples:
        >>> dataset1 = MyCustomDataset(params1)
        >>> dataset2 = MyCustomDataset(params2)
        >>> concat_dataset = ConcatDataset([dataset1, dataset2])
        >>> print(len(concat_dataset))  # Total length of both datasets
        >>> data_point = concat_dataset[1500]  # Accesses an element from the appropriate dataset

    This can also be accomplished by passing in a list of datasets to the YAML config::

        dataset:
          - _component_: torchtune.datasets.instruct_dataset
            source: vicgalle/alpaca-gpt4
            split: train
            train_on_input: True
            packed: True
          - _component_: torchtune.datasets.grammar_dataset
            split: train[:1%]
            train_on_input: False
            packed: True

    This class primarily focuses on providing a unified interface to access elements from multiple datasets,
    enhancing the flexibility in handling diverse data sources for training machine learning models.
    """

    def __init__(self, datasets: list[Dataset]):
        self._datasets: list[Dataset] = datasets

        is_packed = [isinstance(dataset, PackedDataset) for dataset in datasets]
        if any(is_packed) and not all(is_packed):
            raise ValueError(
                "ConcatDataset can't process a mix of packed and non-packed datasets."
            )
        self.packed = all(is_packed)
        self._len: int = sum(len(dataset) for dataset in datasets)
        self._indexes: list[tuple[int, int, int]] = []

        # Calculate distribution of indexes in all datasets
        cumulative_index = 0
        for idx, dataset in enumerate(datasets):
            next_cumulative_index = cumulative_index + len(dataset)
            self._indexes.append((cumulative_index, next_cumulative_index, idx))
            cumulative_index = next_cumulative_index

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        for start, stop, dataset_index in self._indexes:
            if start <= index < stop:
                dataset = self._datasets[dataset_index]
                return dataset[index - start]

    def __len__(self) -> int:
        return self._len
