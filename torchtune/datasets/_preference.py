# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, PromptTemplate

from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform


class PreferenceDataset(Dataset):
    """
    Class that supports any custom dataset with instruction-based prompts and a
    configurable template.

    The general flow from loading a sample to tokenized prompt is:
    load sample -> apply transform -> format into template -> tokenize

    If the column/key names differ from the expected names in the :class:`~torchtune.data.InstructTemplate`,
    then the ``column_map`` argument can be used to provide this mapping.

    Args:
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        message_transform (Transform): callable that keys into the desired fields in the sample
            and converts text content to a list of :class:`~torchtune.data.Message`. It is expected that the final list
            of messages are stored in the ``"messages"`` key.
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        prompt_template (Optional[PromptTemplate]): template used to format the messages based on their role. This is used
            to add structured text around the actual messages. The structured text is used in three scenarios:

            - Task-specific templates to gear models for a particular task that it will expect after training
            - Model-specific templates that are required whenever the model is prompted, such as the [INST]
              tags in Llama2 and in Mistral
            - Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

            The extra text will still get tokenized as normal text, not as special tokens.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``,
            such as ``data_files`` or ``split``.
    """

    def __init__(
        self,
        *,
        source: str,
        message_transform: Transform,
        tokenizer: ModelTokenizer,
        prompt_template: Optional[PromptTemplate] = None,
        filter_fn: Optional[Callable] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._prompt_template = prompt_template
        self._message_transform = message_transform
        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        transformed_sample = self._message_transform(sample)
        if self._prompt_template is not None:
            transformed_sample["chosen"] = self._prompt_template(
                transformed_sample["chosen"]
            )
            transformed_sample["rejected"] = self._prompt_template(
                transformed_sample["rejected"]
            )

        # TODO: Truncation differs from original DPO repo
        # in DPO: first truncate prompts, then responses
        chosen_input_ids, c_masks = self._tokenizer.tokenize_messages(
            transformed_sample["chosen"],
        )
        chosen_labels = list(
            np.where(c_masks, CROSS_ENTROPY_IGNORE_IDX, chosen_input_ids)
        )

        rejected_input_ids, r_masks = self._tokenizer.tokenize_messages(
            transformed_sample["rejected"],
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
