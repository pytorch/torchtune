# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

from torch.utils.data import Dataset

from torchtune import utils

log = utils.get_logger("DEBUG")


class MultiDataset(Dataset):
    def __init__(self, datasets: List[List[Dataset]]):
        self._datasets = datasets
        self._len = sum(len(dataset) for dataset in datasets)
        self._indexes = []

        # Calculate distribution of indexes in all datasets
        cumulative_index = 0
        for idx, dataset in enumerate(datasets):
            next_cumulative_index = cumulative_index + len(dataset)
            self._indexes.append((cumulative_index, next_cumulative_index, idx))
            cumulative_index = next_cumulative_index

        log.debug(f"Datasets summary length: {self._len}")
        log.debug(f"Datasets indexes: {self._indexes}")

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        for start, stop, dataset_index in self._indexes:
            if start <= index < stop:
                dataset = self._datasets[dataset_index]
                return dataset[index - start]  # noqa

    def __len__(self) -> int:
        return self._len
