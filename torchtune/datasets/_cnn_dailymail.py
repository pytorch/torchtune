# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional

from torchtune.datasets._text_completion import TextCompletionDataset

from torchtune.modules.tokenizers import ModelTokenizer


def cnn_dailymail_articles_dataset(
    tokenizer: ModelTokenizer,
    source: str = "ccdv/cnn_dailymail",
    max_seq_len: Optional[int] = None,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> TextCompletionDataset:
    """
    Support for family of datasets similar to `CNN / DailyMail <https://huggingface.co/datasets/ccdv/cnn_dailymail>`_,
    a corpus of news articles. This builder only extracts the articles and not the highlights for
    general text completion tasks.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Returns:
        TextCompletionDataset: the configured TextCompletionDataset
    """

    return TextCompletionDataset(
        tokenizer=tokenizer,
        source=source,
        column="article",
        max_seq_len=max_seq_len,
        filter_fn=filter_fn,
        split=split,
        # This is used to specify the version of the dataset, a required argument
        # by the cnn_dailymail dataset builder:
        # https://huggingface.co/datasets/ccdv/cnn_dailymail/blob/main/cnn_dailymail.py#L80
        name="3.0.0",
        **load_dataset_kwargs,
    )
