# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.config._utils import _get_component_from_path
from torchtune.data import (
    CROSS_ENTROPY_IGNORE_IDX,
    InstructTemplate,
    Message,
    validate_messages,
)
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.tokenizers import ModelTokenizer


class InstructDataset(Dataset):
    """
    Class that supports any custom dataset with instruction-based prompts and a
    configurable template.

    The general flow from loading a sample to tokenized prompt is:
    load sample -> apply transform -> format into template -> tokenize

    If the column/key names differ from the expected names in the :class:`~torchtune.data.InstructTemplate`,
    then the ``column_map`` argument can be used to provide this mapping.

    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is
    set to ``False`` by default.
    - If ``train_on_input`` is True, the prompt is used during training and
    contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        template (InstructTemplate): template used to format the prompt. If the placeholder variable
            names in the template do not match the column/key names in the dataset, use ``column_map`` to map them.
        transform (Optional[Callable]): transform to apply to the sample before formatting to the template.
            Default is None.
        column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names in the template
            to the column/key names in the sample. If None, assume these are identical.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        source: str,
        template: InstructTemplate,
        transform: Optional[Callable] = None,
        column_map: Optional[Dict[str, str]] = None,
        train_on_input: bool = False,
        max_seq_len: Optional[int] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        if not isinstance(template(), InstructTemplate):
            raise ValueError(
                f"template must be an InstructTemplate class, not {type(template())}"
            )

        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self.template = template
        self._transform = transform
        self._column_map = column_map
        self.train_on_input = train_on_input
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        transformed_sample = self._transform(sample) if self._transform else sample

        prompt = self.template.format(transformed_sample, self._column_map)
        key_output = (
            self._column_map["output"]
            if self._column_map and "output" in self._column_map
            else "output"
        )
        messages = [
            Message(role="user", content=prompt, masked=(not self.train_on_input)),
            Message(role="assistant", content=transformed_sample[key_output]),
        ]

        validate_messages(messages)

        tokens, mask = self._tokenizer.tokenize_messages(
            messages, max_seq_len=self.max_seq_len
        )

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, tokens))
        assert len(tokens) == len(labels)

        return {"tokens": tokens, "labels": labels}


def instruct_dataset(
    *,
    tokenizer: ModelTokenizer,
    source: str,
    template: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    max_seq_len: Optional[int] = None,
    packed: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> InstructDataset:
    """
    Build a configurable dataset with instruction prompts. This method should be
    used to configure a custom instruct dataset from the yaml config instead of
    using :class:`~torchtune.datasets.InstructDataset` directly, as it is made to be config friendly.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        template (str): full import path of class used to format the prompt. If the placeholder variable
            names in the template do not match the column/key names in the dataset, use ``column_map`` to map them.
        column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names in the template
            to the column/key names in the sample. If None, assume these are identical.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Examples:
        >>> from torchtune.datasets import instruct_dataset
        >>> dataset = instruct_dataset(
        ...   tokenizer=tokenizer,
        ...   source="yahma/alpaca_cleaned",
        ...   template="torchtune.data.AlpacaInstructTemplate",
        ...   max_seq_len=2096,
        ...   train_on_input=True,
        ...   packed=True,
        ... )

    This can also be accomplished via the yaml config::

        dataset:
            _component_: torchtune.datasets.instruct_dataset
            source: yahma/alpaca_cleaned
            template: torchtune.data.AlpacaInstructTemplate
            max_seq_len: 2096
            train_on_input: True
            packed: True

    Returns:
        InstructDataset or PackedDataset: the configured :class:`~torchtune.datasets.InstructDataset`
            or :class:`~torchtune.datasets.PackedDataset` if ``packed=True``
    """
    ds = InstructDataset(
        tokenizer=tokenizer,
        source=source,
        template=_get_component_from_path(template),
        column_map=column_map,
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )
    return (
        PackedDataset(ds, max_seq_len=max_seq_len, padding_idx=tokenizer.pad_id)
        if packed
        else ds
    )
