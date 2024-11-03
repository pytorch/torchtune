# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional, Union

from torchtune.datasets._packed import PackedDataset

from torchtune.datasets._text_completion import (
    text_completion_dataset,
    TextCompletionDataset,
)

from torchtune.modules.tokenizers import ModelTokenizer


def wikitext_dataset(
    tokenizer: ModelTokenizer,
    source: str = "EleutherAI/wikitext_document_level",
    subset: str = "wikitext-103-v1",
    max_seq_len: Optional[int] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[TextCompletionDataset, PackedDataset]:
    """
    Support for family of datasets similar to `wikitext
    <https://huggingface.co/datasets/EleutherAI/wikitext_document_level>`_,
    an unstructured text corpus consisting of fulls articles from Wikipedia.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        subset (str): name of subset of data to use, see the `wikitext page
            <https://huggingface.co/datasets/EleutherAI/wikitext_document_level#data-instances>`_
            for available subsets. Default is ``"wikitext-103-v1"``.
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Returns:
        Union[TextCompletionDataset, PackedDataset]: the configured :class:`~torchtune.datasets.TextCompletionDataset`
            or :class:`~torchtune.datasets.PackedDataset` if ``packed=True``
    """

    return text_completion_dataset(
        tokenizer=tokenizer,
        source=source,
        column="page",
        max_seq_len=max_seq_len,
        name=subset,
        packed=packed,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )
