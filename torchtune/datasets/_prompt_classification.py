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


class PromptClassificationDataset(Dataset):
    """
    Modified dataset from TextCompletionDataset for classifying any unstructured text corpus. Data format would be
    in a prompt and completion format similar to openai gpt-3 finetuning. Quickly load any dataset
    from Hugging Face or local disk and tokenize it for your model.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        from_disk (bool): Whether to load the dataset from disk. Default is False.
        text_column (str): Name of column in the sample that contains the text data. This is typically required
        label_column (str): Name of column in the sample that contains the label data. Default is "label".
        split (str): Name of the split to load from the dataset. Default is "train".
        classes (Optional[List[Any]]): List of classes for label encoding. If None, the classes will be created from the labels
            in the dataset. Default is None.
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        source: str,
        from_disk: bool = False,
        text_column: str = "text",
        label_column: str = "label",
        split: str = "train",
        classes: Optional[List[Any]] = None,
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
        if classes is None:
            # create classes from labels
            self._label_encoder = {
                label: index
                for index, label in enumerate(
                    sorted(list(set(self._data[self._label_column])))
                )
            }
        else:
            self._label_encoder = {label: index for index, label in enumerate(classes)}
            # check if all labels in dataset have key in label_encoder
            assert set(self._data[self._label_column]).issubset(
                set(self._label_encoder.keys())
            ), "Not all labels in dataset have a key in label_encoder"
        self._label_decoder = {v: k for k, v in self._label_encoder.items()}
        self.num_classes = len(self._label_encoder)

        # One-hot encode the labels into a list
        # self._data['one_hot_labels'] = F.one_hot(
        #     tensor([self._label_encoder[label] for label in self._data[self._label_column]]),
        #     num_classes=len(self._label_encoder),
        # ).squeeze().tolist()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        tokens = self._tokenizer.encode(
            text=sample[self._text_column], add_bos=True, add_eos=True
        )

        # Truncate if needed, but don't coerce EOS id.
        if self.max_seq_len is not None:
            tokens = truncate(tokens, self.max_seq_len - 1)

        # Map labels to classes
        labels = self._label_encoder[sample[self._label_column]]

        return {"tokens": tokens, "labels": labels}


# probably don't need this if we're not doing packed dataset stuff
def prompt_classification_dataset(
    tokenizer: ModelTokenizer,
    source: str,
    from_disk: Optional[bool] = False,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    split: Optional[str] = "train",
    classes: Optional[List[Any]] = None,
    max_seq_len: Optional[int] = None,
    packed: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> PromptClassificationDataset:
    """
    Build a configurable dataset from a freeform, unstructured text corpus similar
    to datasets used in pre-training. This method should be
    used to configure a custom text dataset from the yaml config instead of
    using :class:`~torchtune.datasets.TextClassificationDataset` directly, as it is made to be config friendly.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        from_disk (Optional[bool]): Whether to load the dataset from disk. Default is False.
        text_column (Optional[str]): name of column in the sample that contains the text data. This is typically required
            for Hugging Face datasets or tabular data, but can be omitted for local datasets. Default is None.
        label_column (Optional[str]): name of column in the sample that contains the label data. Default is None.
        split (Optional[str]): Name of the split to load from the dataset. Default is "train".
        classes (Optional[List[Any]]): List of classes for label encoding. If None, the classes will be created from the labels
            in the dataset. Default is None.
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
    ds = PromptClassificationDataset(
        tokenizer=tokenizer,
        source=source,
        from_disk=from_disk,
        text_column=text_column,
        label_column=label_column,
        split=split,
        classes=classes,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )
    return (
        PackedDataset(ds, max_seq_len=max_seq_len, padding_idx=tokenizer.pad_id)
        if packed
        else ds
    )
