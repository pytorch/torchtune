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
        self._len = sum(len(sublist) for sublist in datasets)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        cumulative_index = 0
        for dataset in self._datasets:
            # Let's check if the index is in the dataset
            if cumulative_index + len(dataset) > index:
                # If yes, then prepare the sample and return it
                return dataset[index - cumulative_index]  # noqa
            # Otherwise, go to the next dataset
            cumulative_index += len(dataset)

    def __len__(self) -> int:
        return self._len
