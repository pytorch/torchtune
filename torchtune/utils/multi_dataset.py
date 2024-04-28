# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchtune import config, utils
from tqdm import tqdm

log = utils.get_logger("DEBUG")


class MultiDataset(Dataset):
    def __init__(self, datasets: DictConfig[DictConfig], tokenizer):
        self._datasets = datasets
        self._tokenizer = tokenizer
        self._data = []

        # Load all datasets one by one
        for dataset in datasets:
            self._data.extend(self._loading_dataset(dataset))

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)

    def _loading_dataset(
        self, dataset: DictConfig
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Loads the dataset from provided omegaconf object, and returns it as a list.
        Each element in the list is a tuple of tokens and labels.
        """
        log.info(f"Loading dataset {dataset.source}")
        items = config.instantiate(dataset, tokenizer=self._tokenizer)

        output = []
        for tokens, labels in tqdm(items):
            output.append((tokens, labels))
        return output
