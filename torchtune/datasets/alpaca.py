# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Callable

from datasets import load_dataset, Dataset
from torch.utils.data import Dataset

# Not ideal to import this type here but it's needed for the transform function
from torchtune.models.llama2.tokenizer import Tokenizer


class InstructionTuningDataset(Dataset):
    """PyTorch Representation of an Instruction Fine Tuning Dataset
    from Hugging Face.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.

    Data input format:
    {
        "instruction": "Create a classification task by clustering the given list of items.",
        "input": "Apples, oranges, bananas, strawberries, pineapples",
        "output": "Class 1: Apples, Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",
        "text": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a classification task by clustering the given list of items.\n\n### Input:\nApples, oranges, bananas, strawberries, pineapples\n\n### Response:\nClass 1: Apples,
        Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples",  # noqa: B950
    }

    Example:
    >>> alpaca_ds = AlpacaDataset(tokenizer=tokenizer)
    >>> for batch in Dataloader(alpaca_ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """

    def __init__(self, dataset: Dataset, tokenizer: Tokenizer, row_to_input_and_label: Callable, **kwargs) -> None:
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._row_to_input_and_label = row_to_input_and_label

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        row = self._dataset[index]
        input, label = self._row_to_input_and_label(row)
        return self._tokenizer.encode(input), self._tokenizer.encode(label)
