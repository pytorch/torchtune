# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple
from copy import copy
from datasets import load_dataset
from torch.utils.data import Dataset

# Not ideal to import this type here but it's needed for the transform function
from torchtune.modules import Tokenizer


_CROSS_ENTROPY_IGNORE_IDX = -100


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
        label = [
            _CROSS_ENTROPY_IGNORE_IDX for _ in range(len(inst_inp_response_tag))
        ] + response
        assert len(input) == len(label)
        return input, label

# TODO: this is just a hack to replicate lit-gpt logic for now
class AlpacaCleanedDataset(Dataset):
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
        self._data = load_dataset("yahma/alpaca-cleaned", split="train")
        self._tokenizer = tokenizer

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self._transform(self._data[index])

    def _transform(self, sample: Dict[str, str], mask_inputs: bool = False) -> Tuple[List[int], List[int]]:
        """
        Split a sample on ``response`` tag to create input and labels.

        Args:
            sample (str): Sample text.

        Returns:
            Tuple of encoded inputs and labels.
        """
        full_prompt = generate_prompt(sample)
        full_prompt_and_response = full_prompt + sample["output"]

        encoded_full_prompt = self._tokenizer.encode(full_prompt, add_bos=True, add_eos=False)
        # their use_bos is True so should have add_bos = True, eos default is False
        # todo: why don't we handle max len
        # encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)

        # they have both bos (cause of default) and eos (explicitly passed) as true here
        encoded_full_prompt_and_response = self._tokenizer.encode(full_prompt_and_response)
        # encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)


        # The labels are the full prompt with response, but with the prompt masked out
        labels = copy(encoded_full_prompt_and_response)
        if mask_inputs:
            labels[: len(encoded_full_prompt)] = _CROSS_ENTROPY_IGNORE_IDX

        return encoded_full_prompt_and_response, labels




def generate_prompt(example: Dict[str, str]) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )
