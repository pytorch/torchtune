# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

from torchtune.datasets._verifiable import PromptToMessage, VerifiableDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer


def calc_gsm8k_dataset(
    tokenizer: ModelTokenizer,
    source: str = "MU-NLPC/Calc-gsm8k",
    column_map: Optional[Dict[str, str]] = None,
    new_system_prompt: Optional[str] = None,
    split: str = "train",
    **load_dataset_kwargs: Any,
):
    """
    Builds the Calc-GSM8k dataset using TorchTune's PreferenceDataset utilities.

    Args:
        tokenizer (ModelTokenizer): The tokenizer to preprocess the data.
        source (str): The underlying dataset
        column_map (Optional[Dict[str, str]]): list of columns to include, must include prompt
        new_system_prompt (Optional[str]): An additional system prompt to add in front of the prompt
        split (str): which dataset split to use, defaults to 'train'
        **load_dataset_kwargs (Any): arguments to be passed through for dataset loading

    Returns:
        Dataset: A TorchTune-compatible dataset (VerifiableDataset).
    """
    if column_map is None:
        column_map = {"prompt": "question", "result": "result"}

    ds = VerifiableDataset(
        source=source,
        tokenizer=tokenizer,
        message_transform=PromptToMessage(
            column_map=column_map,
            new_system_prompt=new_system_prompt,
        ),
        split=split,
        **load_dataset_kwargs,
    )

    return ds
