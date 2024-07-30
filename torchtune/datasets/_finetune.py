# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, PromptTemplate
from torchtune.modules.transforms import Transform


class FinetuneDataset(Dataset):
    """
    Dataset class for creating instruct, chat, tool, or multimodal datasets for fine-tuning.

    All datasets can be considered "conversations" with the model, or AI assistant.
    Thus, we can format all text content as messages in a conversation assigned to
    a :class:`~torchtune.data.Role`:

    - system messages contain the system prompt
    - user messages contain the input prompt into the model
    - assistant messages are the response of the model and what you actually want
      to train for and compute loss directly against
    - ipython messages are the return from a tool call

    Chat datasets are multiple rounds of user-assistant messages. Instruct datasets
    are typically a single round involving a specific instruction and the model's response.

    The :class:`~torchtune.data.Message` forms the core data unit that all tokenizer
    APIs expect. The key component of this class that ensures any dataset is transformed
    into thie format is the ``message_transform``. This is a callable class that takes
    in a sample dictionary - typically a single row from a Hugging Face dataset or a single
    json - that processes the sample in any configurable way to output a list of messages::

        [
            Message(
                role=<system|user|assistant|ipython>,
                content=<message>,
            ),
            ...
        ]

    For any custom dataset, use the ``message_transform`` to contain all pre-processing to
    return the list of messages.

    Any model specific pre-processing that needs to happen can be configured with the ``model_transform``
    parameter. This is another callable class that contains any custom logic tied to the
    model you are fine-tuning. For example, text + image multimodal datasets requires processing
    the images in a way specific to the vision encoder being used by the model and is agnostic
    to the specific dataset.

    Tokenization is handled by the ``model_transform``. All :class:`~torchtune.modules.tokenizers.ModelTokenizer`s
    can be treated as a ``model_transform`` since it uses the model-specific tokenizer to
    transform the list of messages outputted from the ``message_transform`` into tokens
    used by the model for training. Text-only datasets will simply pass the :class:`~torchtune.modules.tokenizers.ModelTokenizer`
    into ``model_transform``.

    The general pipeline is then: raw sample -> optional filter -> apply dataset-specific message transform -> apply
    optional prompt template -> apply model-specific transform -> tokens used for training

    Args:
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        message_transform (Transform): callable that keys into the desired fields in the sample
            and converts text content to a list of :class:`~torchtune.data.Message`. It is expected that the final list
            of messages are stored in the ``"messages"`` key.
        model_transform (Transform): callable that applies model-specific pre-processing to the sample after the list of
            messages is created from ``message_transform``. This includes tokenization and any modality-specific
            transforms.
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
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.
    """

    def __init__(
        self,
        *,
        source: str,
        message_transform: Transform,
        model_transform: Transform,
        prompt_template: Optional[PromptTemplate] = None,
        filter_fn: Optional[Callable] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._message_transform = message_transform
        self._prompt_template = prompt_template
        self._model_transform = model_transform

        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        transformed_sample = self._message_transform(sample)
        if self._prompt_template is not None:
            transformed_sample["messages"] = self._prompt_template(
                transformed_sample["messages"]
            )
        tokenized_dict = self._model_transform(transformed_sample)

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        tokenized_dict["labels"] = list(
            np.where(
                tokenized_dict["mask"],
                CROSS_ENTROPY_IGNORE_IDX,
                tokenized_dict["tokens"],
            )
        )
        assert len(tokenized_dict["tokens"]) == len(tokenized_dict["labels"])

        return tokenized_dict
