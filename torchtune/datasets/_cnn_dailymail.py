# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

from torchtune.datasets._text_completion import TextCompletionDataset

from torchtune.modules.tokenizers import ModelTokenizer


def cnn_dailymail_articles_dataset(
    tokenizer: ModelTokenizer,
    source: str = "ccdv/cnn_dailymail",
    max_seq_len: Optional[int] = None,
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
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Returns:
        TextCompletionDataset: the configured TextCompletionDataset
    """

    return TextCompletionDataset(
        tokenizer=tokenizer,
        source=source,
        column="article",
        max_seq_len=max_seq_len,
        split="train",
        # This is used to specify the version of the dataset, a required argument
        # by the cnn_dailymail dataset builder:
        # https://huggingface.co/datasets/ccdv/cnn_dailymail/blob/main/cnn_dailymail.py#L80
        name="3.0.0",
        **load_dataset_kwargs,
    )
