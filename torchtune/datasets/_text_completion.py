# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Mapping, Optional

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data import truncate
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.tokenizers import ModelTokenizer


class TextCompletionDataset(Dataset):
    """
    Freeform dataset for any unstructured text corpus. Quickly load any dataset
    from Hugging Face or local disk and tokenize it for your model.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        column (str): name of column in the sample that contains the text data. This is typically required
            for Hugging Face datasets or tabular data. For local datasets with a single column, use the default "text",
            which is what is assigned by Hugging Face datasets when loaded into memory. Default is "text".
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        source: str,
        column: str = "text",
        max_seq_len: Optional[int] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self.max_seq_len = max_seq_len
        self._column = column

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        prompt = sample[self._column]
        tokens = self._tokenizer.encode(text=prompt, add_bos=True, add_eos=True)

        # Truncate if needed, but don't coerce EOS id
        if self.max_seq_len is not None:
            tokens = truncate(tokens, self.max_seq_len - 1)

        # No need to offset labels by 1 - happens in the recipe
        labels = tokens.copy()

        return {"tokens": tokens, "labels": labels}


def text_completion_dataset(
    tokenizer: ModelTokenizer,
    source: str,
    column: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    packed: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> TextCompletionDataset:
    """
    Build a configurable dataset from a freeform, unstructured text corpus similar
    to datasets used in pre-training. This method should be
    used to configure a custom text dataset from the yaml config instead of
    using :class:`~torchtune.datasets.TextCompletionDataset` directly, as it is made to be config friendly.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        column (Optional[str]): name of column in the sample that contains the text data. This is typically required
            for Hugging Face datasets or tabular data, but can be omitted for local datasets. Default is None.
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Examples:
        >>> from torchtune.datasets import text_completion_dataset
        >>> dataset = text_completion_dataset(
        ...   tokenizer=tokenizer,
        ...   source="allenai/c4",
        ...   column="text",
        ...   max_seq_len=2096,
        ...   data_dir="realnewslike",
        ...   packed=False,
        ... )

    This can also be accomplished via the yaml config::

        dataset:
            _component_: torchtune.datasets.text_completion_dataset
            source: allenai/c4
            column: text
            max_seq_len: 2096
            data_dir: realnewslike
            packed: False

    Returns:
        TextCompletionDataset or PackedDataset: the configured :class:`~torchtune.datasets.TextCompletionDataset`
            or :class:`~torchtune.datasets.PackedDataset` if ``packed=True``
    """
    ds = TextCompletionDataset(
        tokenizer=tokenizer,
        source=source,
        column=column,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )
    return (
        PackedDataset(ds, max_seq_len=max_seq_len, padding_idx=tokenizer.pad_id)
        if packed
        else ds
    )
