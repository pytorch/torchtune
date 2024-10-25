# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Based off of TextCompletionDatset in _text_completion.py from torchtune

from typing import Any, Dict, List, Mapping, Optional

from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from torchtune.data import truncate
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.tokenizers import ModelTokenizer


class PromptCompletionDataset(Dataset):
    """
    Modified dataset from TextCompletionDataset for classifying any unstructured text corpus. Data format would be
    in a prompt and completion format similar to openai gpt-3 finetuning. Quickly load any dataset
    from Hugging Face or local disk and tokenize it for your model.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): Path string of dataset, anything supported by Hugging Face's ``load_dataset`` or a local path.
        from_disk (bool): If True, load dataset from local disk using ``load_from_disk``. Default is False.
        text_column (str): Name of column containing the prompt text. Default is "text".
        label_column (str): Name of column containing the label/completion text. Default is "label".
        split (str): Which split of the dataset to use. Default is "train".
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input sequence.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        **load_dataset_kwargs (Dict[str, Any]): Additional keyword arguments to pass to ``load_dataset`` or ``load_from_disk``.
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        source: str,
        from_disk: bool = False,
        text_column: str = "text",
        label_column: str = "label",
        split: str = "train",
        max_seq_len: Optional[int] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        if from_disk:
            self._data = load_from_disk(source, **load_dataset_kwargs)[split]
        else:
            self._data = load_dataset(source, **load_dataset_kwargs)
        self.max_seq_len = max_seq_len
        self._text_column = text_column
        self._label_column = label_column
        # Add label separator
        self._label_separator = self._tokenizer.encode(
            text="]\n\n## Label:\n",
            # text="## Label:\n",
            add_bos=False,
            add_eos=False,
            trim_leading_whitespace=True,
        )
        self._label_separator_len = len(self._label_separator)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        tokens = self._tokenizer.encode(
            text=sample[self._text_column], add_bos=True, add_eos=False
        )

        # Truncate if needed, but leave room for label separator.
        if self.max_seq_len is not None:
            tokens = truncate(tokens, self.max_seq_len - self._label_separator_len)

        # Add label separator
        tokens.extend(self._label_separator)

        # Directly state the label_column field
        labels = self._tokenizer.encode(
            sample[self._label_column].strip(), add_bos=False, add_eos=True
        )

        labels = [-100] * len(tokens) + labels

        return_dict = {"tokens": tokens, "labels": labels}
        if "sample_weights" in sample:
            return_dict["sample_weights"] = sample["sample_weights"]

        return return_dict


def prompt_completion_dataset(
    tokenizer: ModelTokenizer,
    source: str,
    from_disk: Optional[bool] = False,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    split: Optional[str] = "train",
    max_seq_len: Optional[int] = None,
    packed: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> PromptCompletionDataset:
    """
    Build a configurable dataset from a freeform, unstructured text corpus similar
    to datasets used in pre-training. This method should be
    used to configure a custom text dataset from the yaml config instead of
    using :class:`~torchtune.datasets.TextClassificationDataset` directly, as it is made to be config friendly.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        from_disk (Optional[bool]): If True, load dataset from local disk using ``load_from_disk``. Default is False.
        text_column (Optional[str]): name of column in the sample that contains the text data. This is typically required
            for Hugging Face datasets or tabular data, but can be omitted for local datasets. Default is None.
        label_column (Optional[str]): name of column in the sample that contains the label data. Default is None.
        split (Optional[str]): Which split of the dataset to use. Default is "train".
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Examples:
        >>> from torchtune.datasets import text_classification_dataset
        >>> dataset = text_completion_dataset(
        ...   tokenizer=tokenizer,
        ...   source="stanfordnlp/imdb",
        ...   text_column="text",
        ...   label_column="label",
        ...   max_seq_len=2096,
        ...   data_dir="imdb_finetuned",
        ...   packed=False,
        ... )

    This can also be accomplished via the yaml config::

        dataset:
            _component_: torchtune.datasets.text_classification_dataset
            source: stanfordnlp/imdb
            text_column: text
            label_column: label
            max_seq_len: 2096
            data_dir: imdb_finetuned
            packed: False

    Returns:
        TextClassificationDataset or PackedDataset: the configured :class:`~torchtune.datasets.TextCompletionDataset`
            or :class:`~torchtune.datasets.PackedDataset` if ``packed=True``
    """
    ds = PromptCompletionDataset(
        tokenizer=tokenizer,
        source=source,
        from_disk=from_disk,
        text_column=text_column,
        label_column=label_column,
        split=split,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )
    return (
        PackedDataset(ds, max_seq_len=max_seq_len, padding_idx=tokenizer.pad_id)
        if packed
        else ds
    )
