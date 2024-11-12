# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional

from torchtune.data import ChosenRejectedToMessages
from torchtune.datasets._preference import PreferenceDataset
from torchtune.modules.tokenizers import ModelTokenizer


def hh_rlhf_helpful_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "RLHFlow/HH-RLHF-Helpful-standard",
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> PreferenceDataset:
    """
    Constructs preference datasets similar to `Anthropic's helpful/harmless RLHF
    data
    <https://huggingface.co/datasets/RLHFlow/HH-RLHF-Helpful-standard>`_. This is
    the processed helpful subset of the original dataset in a standardized format.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details. Default is ``RLHFlow/HH-RLHF-Helpful-standard``.
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns "chosen" and "rejected"
            in the message transform :class:`~torchtune.data.ChosenRejectedToMessages` to the new column names in
            the dataset. Keys should be "chosen" and "rejected" and values should be the actual column names.
            If None, keep the default columns "chosen" and "rejected".
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        new_system_prompt (Optional[str]): if specified, prepend a system message to every sample for both chosen
            and rejected. This can serve as instructions to guide the model response. Setting this will OVERRIDE
            any system messages already present in the dataset. Default is None.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Returns:
        PreferenceDataset: The preference dataset built from source paired data.
    """

    message_transform = ChosenRejectedToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )

    return PreferenceDataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )
