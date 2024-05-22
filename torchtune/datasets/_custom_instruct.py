# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from torchtune.datasets._instruct import instruct_dataset, InstructDataset
from torchtune.modules.tokenizers import Tokenizer


def custom_instruct_dataset(
    tokenizer: Tokenizer,
    *,
    dataset_path: str,
    train_on_input: bool = True,
    max_seq_len: int = 512,
    packed: bool = False,
) -> InstructDataset:
    """
    Custom dataset loader for Alpaca-style datasets from a local path.

    Masking of the prompt during training is controlled by the `train_on_input` flag, which is
    set to `True` by default.
    - If `train_on_input` is True, the prompt is used during training and
    contributes to the loss.
    - If `train_on_input` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenizer must implement an `encode` and `decode` method.
        dataset_path (str): Path to the local directory where the dataset is saved in JSON format.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is True.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
            Default is 512, but we recommend setting this to the highest you can fit in memory and
            is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.

    Returns:
        InstructDataset: dataset configured with source data and template

    Example:
        >>> custom_ds = custom_dataset(tokenizer=tokenizer, dataset_path="path/to/saved/dataset.json")
        >>> for batch in DataLoader(custom_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    return instruct_dataset(
        tokenizer=tokenizer,
        source=dataset_path,
        template="AlpacaInstructTemplate",
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        packed=packed,
        split="train",
    )


# Example usage:
custom_cleaned_dataset = partial(
    custom_instruct_dataset, dataset_path="path/to/save/dataset.json"
)
