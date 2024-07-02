# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

from torchtune.datasets._text_completion import TextCompletionDataset

from torchtune.modules.tokenizers import ModelTokenizer


def wikitext_dataset(
    tokenizer: ModelTokenizer,
    source: str = "wikitext",
    subset: str = "wikitext-103-raw-v1",
    max_seq_len: Optional[int] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> TextCompletionDataset:
    """
    Support for family of datasets similar to `wikitext <https://huggingface.co/datasets/wikitext>`_,
    an unstructured text corpus consisting of articles from Wikipedia.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        subset (str): name of subset of data to use, see the `wikitext page <https://huggingface.co/datasets/wikitext#data-fields>`_
            for available subsets.
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
        column="text",
        max_seq_len=max_seq_len,
        name=subset,
        split="train",
        **load_dataset_kwargs,
    )
