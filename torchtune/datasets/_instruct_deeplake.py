# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import deeplake
import numpy as np
from torch.utils.data import Dataset
from torchtune.config._utils import _get_instruct_template

from torchtune.data import (
    CROSS_ENTROPY_IGNORE_IDX,
    InstructTemplate,
    Message,
    validate_messages,
)

from torchtune.modules import Tokenizer

os.environ["ACTIVELOOP_TOKEN"] = os.getenv("ACTIVELOOP_TOKEN")


class DeepLakePyTorchDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        question = self.ds.question[idx].text().astype(str)
        instruction = self.ds.instruction[idx].text().astype(str)
        oracle_context = self.ds.oracle_context[idx].text().astype(str)
        return {
            "question": question,
            "instruction": instruction,
            "oracle_context": oracle_context,
        }


def load_deeplake_dataset(source):
    ds = deeplake.dataset(source)
    return ds
    return DeepLakePyTorchDataset(ds)


class InstructDatasetDeepLakeRAFT(Dataset):
    """
    TODO: Add docstring
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        source: str,
        template: InstructTemplate,
        transform: Optional[Callable] = None,
        column_map: Optional[Dict[str, str]] = None,
        train_on_input: bool = False,
        max_seq_len: Optional[int] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_deeplake_dataset(source)
        self.template = template
        self._transform = transform
        self._column_map = column_map
        self.train_on_input = train_on_input
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        transformed_sample = self._transform(sample) if self._transform else sample

        prompt = self.template.format(transformed_sample, self._column_map)
        key_output = (
            self._column_map["oracle_context"]
            if self._column_map and "oracle_context" in self._column_map
            else "oracle_context"
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

        return tokens, labels


def instruct_dataset(
    tokenizer: Tokenizer,
    source: str,
    template: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    max_seq_len: Optional[int] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> InstructDatasetDeepLakeRAFT:
    """
    Build a configurable dataset with instruction prompts. This method should be
    used to configure a custom instruct dataset from the yaml config instead of
    using `InstructDataset` directly, as it is made to be config friendly.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by Activeloop
        template (str): class used to format the prompt. If the placeholder variable
            names in the template do not match the column/key names in the dataset, use `column_map` to map them.
        column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names in the template
            to the column/key names in the sample. If None, assume these are identical.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to `load_dataset`.

    Returns:
        InstructDataset: the configured InstructDataset
    """
    return InstructDatasetDeepLakeRAFT(
        tokenizer=tokenizer,
        source=source,
        template=_get_instruct_template(template),
        column_map=column_map,
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        **load_dataset_kwargs,
    )
