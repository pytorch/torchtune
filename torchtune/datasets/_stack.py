# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from torchtune.datasets._concat import ConcatDataset
from torchtune.modules.tokenizers import Tokenizer


def stack_dataset(
    tokenizer: Tokenizer,
    source: str = "bigcode/the-stack-dedup",
    text_column: str = "content",
    max_seq_len: int = 4096,
    max_rows: int = 1000,
    is_local: bool = False,
    shuffle_before_packing: bool = True,
    seed: int = 29,
    train_split_name: str = "train",
    train_on_input: bool = None,  # dummy argument for compatibility with config yaml files
    **load_dataset_kwargs: Dict[str, Any],
) -> ConcatDataset:
    """
    Example showing how to pass the [The Stack V1 Dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup)
    as a streaming dataset to the ConcatDataset class for continued pretraining. The Stack Dedup is
    a dataset of code that is a popular solution for training code assistants.

    Note how `streaming` and `token` values are passed to the `load_dataset_kwargs` dictionary - they are not
    function arguments. This example uses the `HF_TOKEN` environment variable to authenticate
    with the Hugging Face API. If you don't have that env var set, you can pass your token directly as a string.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`,
            or if `is_local` is true, a local path to load with `load_from_disk`.
        text_column (str): column name of the dataset to pack into samples.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
            Default is 512, but we recommend setting this to the highest you can fit in memory and
            is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        max_rows (int): maximum number of samples to pack. Default is 1000. Note: The Stack Dedup is very large,
            so we limit the number of samples to 1000 for this example.
        is_local (bool): whether the source dataset is a local path. Default is False.
        shuffle_before_packing (bool): whether to shuffle the dataset before packing. Default is True.
        seed (int): seed for shuffling. Default is 29.
        train_split_name (str): name of the split to use. Default is 'train'.
        train_on_input (bool): dummy argument for compatibility with config yaml files.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to `load_dataset`. You will need
            to pass a Huggingface API token value as a `token`, and we recommend setting `streaming` to True, since the
            stack is a large dataset.

    Returns:
        ConcatDataset: dataset containing packed, tokenized samples.


    Example:
        >>> stack_ds = stack_dataset(tokenizer=tokenizer, max_seq_len=2048, max_rows=50000)
        >>> for batch in Dataloader(stack_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    return ConcatDataset(
        tokenizer=tokenizer,
        source=source,
        text_column=text_column,
        max_seq_len=max_seq_len,
        max_rows=max_rows,
        is_local=is_local,
        shuffle_before_packing=shuffle_before_packing,
        seed=seed,
        train_split_name=train_split_name,
        **load_dataset_kwargs,
    )
