# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.config._errors import InstantiationError
from torchtune.config._utils import _get_component_from_path

from torchtune.data import PromptTemplate
from torchtune.datasets._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules import Tokenizer


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
        template (PromptTemplate): template used to format the prompt. If the placeholder variable
            names in the template do not match the column/key names in the dataset, use `column_map` to map them.
        transform (Optional[Callable]): transform to apply to the sample before formatting to the template.
            Default is None.
        column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names in the template
            to the column/key names in the sample. If None, assume these are identical.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to `load_dataset`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        source: str,
        template: PromptTemplate,
        transform: Optional[Callable] = None,
        column_map: Optional[Dict[str, str]] = None,
        train_on_input: bool = False,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self.template = template
        self._transform = transform
        self._column_map = column_map
        self.train_on_input = train_on_input

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        transformed_sample = self._transform(sample) if self._transform else sample

        prompt = self.template.format(transformed_sample, self._column_map)
        key_output = self._column_map["output"] if self._column_map else "output"
        prompt_with_response = prompt + sample[key_output]

        encoded_prompt = self._tokenizer.encode(
            text=prompt, add_bos=True, add_eos=False
        )
        encoded_prompt_with_response = self._tokenizer.encode(
            text=prompt_with_response, add_bos=True, add_eos=True
        )
        labels = copy.deepcopy(encoded_prompt_with_response)

        if not self.train_on_input:
            labels[: len(encoded_prompt)] = [CROSS_ENTROPY_IGNORE_IDX] * len(
                encoded_prompt
            )

        assert len(encoded_prompt_with_response) == len(labels)

        return encoded_prompt_with_response, labels


def instruct_dataset(
    tokenizer: Tokenizer,
    source: str,
    template: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> InstructDataset:
    """
    Build a configurable dataset with instruction prompts. This method should be
    used to configure a custom instruct dataset from the yaml config instead of
    using `InstructDataset` directly, as it is made to be config friendly.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by HuggingFace's `load_dataset`
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        template (str): class name of template used to format the prompt. If the placeholder variable
            names in the template do not match the column/key names in the dataset, use `column_map` to map them.
        column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names in the template
            to the column/key names in the sample. If None, assume these are identical.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to `load_dataset`.

    Returns:
        InstructDataset: the configured InstructDataset
    """
    return InstructDataset(
        tokenizer=tokenizer,
        source=source,
        template=_get_template(template),
        column_map=column_map,
        train_on_input=train_on_input,
        **load_dataset_kwargs,
    )


def _get_template(template: str) -> PromptTemplate:
    """
    Get the prompt template class from the template string.

    String should either be the PromptTemplate class name directly, or a raw
    string with 1 or more placeholders. If none of these apply, then raise an
    error.

    Args:
        template (str): class name of template, or string with placeholders

    Returns:
        PromptTemplate: the prompt template class or the same verified string

    Raises:
        ValueError: if the template is not a PromptTemplate class or a proper
            template string
    """
    path = "torchtune.data." + template
    try:
        template_class = _get_component_from_path(path)
        return template_class()
    except InstantiationError:
        # Verify that string can be used as a template, should have variable
        # placeholders
        pattern = r"\{.+?\}"
        if not re.search(pattern, template):
            raise ValueError(
                f"Invalid template '{template}': "
                + "Must be a PromptTemplate class or a string with placeholders."
            ) from None
        return template
