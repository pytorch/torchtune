# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import InstructTemplate, tokenize_preference_sample
from torchtune.modules import Tokenizer


class PreferenceDataset(Dataset):
    """
    Class that supports any custom dataset with instruction-based prompts and a
    configurable template.

    The general flow from loading a sample to tokenized prompt is:
    load sample -> apply transform -> format into template -> tokenize

    If the column/key names differ from the expected names in the `InstructTemplate`,
    then the `column_map` argument can be used to provide this mapping.

    Masking of the prompt during training is controlled by the `train_on_input` flag, which is
    set to `False` by default.
    - If `train_on_input` is True, the prompt is used during training and
    contributes to the loss.
    - If `train_on_input` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        template (PromptTemplate): template used to format the prompt. If the placeholder variable
            names in the template do not match the column/key names in the dataset, use `column_map` to map them.
        transform (Optional[Callable]): transform to apply to the sample before formatting to the template.
            Default is None.
        column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names in the template
            to the column/key names in the sample. If None, assume these are identical.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to `load_dataset`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
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

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        transformed_sample = self._transform(sample) if self._transform else sample
        prompt, chosen, rejected = self.template.format(transformed_sample, self._column_map)

        batch = tokenize_preference_sample(
            tokenizer=self._tokenizer,
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            max_length=self.max_seq_len,
        )
        return batch
