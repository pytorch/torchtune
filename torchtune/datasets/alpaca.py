# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

# Not ideal to import this type here but it's needed for the transform function
from torchtune.modules import Tokenizer


class AlpacaDataset(Dataset):
    """
    PyTorch Representation of the Alpaca Dataset
    https://huggingface.co/datasets/tatsu-lab/alpaca
    from Hugging Face.

    Data input format: https://huggingface.co/datasets/tatsu-lab/alpaca#data-instances


    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        **kwargs: Additional keyword arguments to pass to the Alpaca Dataset.


    Example:
        >>> alpaca_ds = AlpacaDataset(tokenizer=tokenizer)
        >>> for batch in Dataloader(alpaca_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    def __init__(self, tokenizer: Tokenizer, **kwargs) -> None:
        self._data = load_dataset("tatsu-lab/alpaca", split="train")
        self._tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self._transform(self._data[index]["text"])

    def _transform(self, sample: str) -> Tuple[List[int], List[int]]:
        """
        Split a sample on ``response`` tag to create input and labels.

        Args:
            sample (str): Sample text.

        Returns:
            Tuple of encoded inputs and labels.
        """
        response_tag = "\n\n### Response:\n"
        inst_inp_response_tag = sample[: sample.index(response_tag) + len(response_tag)]
        response = sample[sample.index(response_tag) + len(response_tag) :]
        inst_inp_response_tag = self._tokenizer.encode(
            inst_inp_response_tag, add_bos=True, add_eos=False
        )
        response = self._tokenizer.encode(response, add_bos=False, add_eos=True)
        input = inst_inp_response_tag + response
        label = [-100 for _ in range(len(inst_inp_response_tag))] + response
        assert len(input) == len(label)
        return input, label
