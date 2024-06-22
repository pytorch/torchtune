# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

from torchtune.config._utils import _get_component_from_path
from torchtune.data import ColumnMap, QuickTemplate
from torchtune.datasets._chat import ChatDataset
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.tokenizers import Tokenizer
from torchtune.modules.transforms import Compose


def instruct_dataset(
    *,
    tokenizer: Tokenizer,
    source: str,
    template: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    max_seq_len: Optional[int] = None,
    packed: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> ChatDataset:
    """
    Build a configurable dataset with instruction prompts. This method should be
    used to configure a custom instruct dataset from the yaml config instead of
    using `InstructDataset` directly, as it is made to be config friendly.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        template (str): full import path of class used to format the prompt. If the placeholder variable
            names in the template do not match the column/key names in the dataset, use `column_map` to map them.
        column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names in the template
            to the column/key names in the sample. If None, assume these are identical.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to `load_dataset`.

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
    transforms = []
    if column_map is not None:
        transforms.append(ColumnMap(column_map))

    if "{content}" in template:
        prompt_template = QuickTemplate(
            template=template, train_on_input=train_on_input
        )
    else:
        prompt_template = _get_component_from_path(template)()

    transforms.append(prompt_template)

    ds = ChatDataset(
        tokenizer=tokenizer,
        source=source,
        message_transform=Compose(transforms),
        max_seq_len=max_seq_len,
        train_on_input=train_on_input,
        **load_dataset_kwargs,
    )
    return PackedDataset(ds, max_seq_len=max_seq_len) if packed else ds
