# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, InstructTemplate, Message

from torchtune.modules.tokenizers import ModelTokenizer


class PreferenceDataset(Dataset):
    """
    Class that supports any custom dataset with instruction-based prompts and a
    configurable template.

    The general flow from loading a sample to tokenized prompt is:
    load sample -> apply transform -> format into template -> tokenize

    If the column/key names differ from the expected names in the :class:`~torchtune.data.InstructTemplate`,
    then the ``column_map`` argument can be used to provide this mapping.

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
        max_seq_len: Optional[int] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self.template = template
        self._transform = transform
        self._column_map = column_map
        self.max_seq_len = max_seq_len
        self._data = self._data.filter(
            lambda x: len(x[column_map["prompt"]]) + len(x[column_map["chosen"]])
            <= max_seq_len
            and len(x[column_map["prompt"]]) + len(x[column_map["rejected"]])
            <= max_seq_len
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        transformed_sample = self._transform(sample) if self._transform else sample
        prompt = self.template.format(transformed_sample, self._column_map)

        column_map = self._column_map or {}
        key_chosen = column_map.get("chosen", "chosen")
        key_rejected = column_map.get("rejected", "rejected")

        chosen_message = [
            Message(role="user", content=prompt, masked=True),
            Message(role="assistant", content=transformed_sample[key_chosen]),
        ]

        rejected_message = [
            Message(role="user", content=prompt, masked=True),
            Message(role="assistant", content=transformed_sample[key_rejected]),
        ]

        # TODO: Trunction differs from original DPO repo
        # in DPO: first truncate prompts, then responses
        chosen_input_ids, c_masks = self._tokenizer.tokenize_messages(
            chosen_message, self.max_seq_len
        )
        chosen_labels = list(
            np.where(c_masks, CROSS_ENTROPY_IGNORE_IDX, chosen_input_ids)
        )

        rejected_input_ids, r_masks = self._tokenizer.tokenize_messages(
            rejected_message, self.max_seq_len
        )
        rejected_labels = list(
            np.where(r_masks, CROSS_ENTROPY_IGNORE_IDX, rejected_input_ids)
        )

        assert len(chosen_input_ids) == len(chosen_labels)
        assert len(rejected_input_ids) == len(rejected_labels)

        batch = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_labels=rejected_labels,
        )

        return batch
