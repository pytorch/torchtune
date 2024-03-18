# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import PromptTemplate

from torchtune.modules import Tokenizer

CROSS_ENTROPY_IGNORE_IDX = -100


class InstructDataset(Dataset):
    """
    Class that supports any custom dataset with instruction-based prompts and a
    configurable template.

    The general flow from loading a sample to tokenized prompt is:
    load sample -> apply transform -> format into template -> tokenize

    If the column/key names differ from the expected names in the `PromptTemplate`,
    then the `column_map` argument can be used to provide this mapping.

    Masking of the prompt during training is controlled by the `train_on_input` flag, which is
    set to `False` by default.
    - If `train_on_input` is True, the prompt is used during training and
    contributes to the loss.
    - If `train_on_input` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by HuggingFace's `load_dataset`
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        template (Union[PromptTemplate, str]): template used to format the prompt. It should be either a `PromptTemplate` or
            an string with placeholders for the appropriate data fields. If the placeholder
            names do not match the column/key names in the dataset, use `column_map` to map them.
        transform (Optional[Callable]): transform to apply to the sample before formatting to the template.
            Default is None.
        column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names in the template
            to the column/key names in the sample. If None, assume these are identical.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        source: str,
        template: Union[PromptTemplate, str],
        transform: Optional[Callable] = None,
        column_map: Optional[Dict[str, str]] = None,
        train_on_input: bool = False,
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source)
        self.template = template
        self._transform = transform
        self._column_map = column_map
        self.train_on_input = train_on_input

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        sample = self._data[index]
        return self._process(sample)

    def _process(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        transformed_sample = self._transform(sample) if self._transform else sample

        prompt = self.template.format(transformed_sample, self._column_map)

        encoded_prompt = self._tokenizer.encode(prompt)

        key_output = self._column_map["output"] if self._column_map else "output"
        response = self._tokenizer.encode(sample[key_output])

        encoded_prompt_with_response = encoded_prompt + response

        if self.train_on_input:
            labels = copy.deepcopy(encoded_prompt_with_response)
        else:
            labels = [CROSS_ENTROPY_IGNORE_IDX] * len(encoded_prompt) + response

        return encoded_prompt_with_response, labels
