# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.config._utils import _get_component_from_path
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, Message
from torchtune.modules.transforms import Transform


class FinetuneDataset(Dataset):
    """
    Class that supports any custom dataset with multiturn conversations.

    The general flow from loading a sample to tokenized prompt is:
    load sample -> apply transform -> foreach turn{format into template -> tokenize}

    If the column/key names differ from the expected names in the ``ChatFormat``,
    then the ``column_map`` argument can be used to provide this mapping.

    Use ``convert_to_messages`` to prepare your dataset into the Llama2 chat format
    and roles::

        [
            Message(
                role=<system|user|assistant>,
                content=<message>,
            ),
            ...
        ]

    This class supports multi-turn conversations. If a tokenizer sample with multiple
    turns does not fit within ``max_seq_len`` then it is truncated.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an ``encode`` and ``decode`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        convert_to_messages (Callable[[Mapping[str, Any]], List[Message]]): function that keys into the desired field in the sample
            and converts to a list of :class:`~torchtune.data.Message` that follows the Llama format with the expected keys
        chat_format (Optional[ChatFormat]): template used to format the chat. This is used to add structured text around the actual
            messages, such as the [INST] tags in Llama2 and in Mistral. The extra text will still get tokenized as normal text, not
            as special tokens. In models like Llama3 where the tokenizer adds tags as special tokens, ``chat_format`` is not needed,
            unless you want to structure messages in a particular way for inference. If the placeholder variable names in the
            template do not match the column/key names in the dataset, use ``column_map`` to map them. For a list of all possible
            chat formats, check out :ref:`chat_formats`. Default: None.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.
    """

    def __init__(
        self,
        *,
        source: str,
        message_transform: Callable[[Mapping[str, Any]], List[Message]],
        model_transform: Transform,
        column_map: Optional[Dict[str, str]] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:

        self._model_transform = model_transform
        self._data = load_dataset(source, **load_dataset_kwargs)
        self._message_transform = message_transform
        self.column_map = column_map

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        if self.column_map is not None:
            for key, value in self.column_map.items():
                sample[value] = sample.pop(key)
        messages = self._message_transform(sample)
        sample.update({"messages": messages})
        prepared_sample = self._model_transform(**sample)
        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        prepared_sample["labels"] = list(
            np.where(
                prepared_sample["mask"],
                CROSS_ENTROPY_IGNORE_IDX,
                prepared_sample["tokens"],
            )
        )
        assert len(prepared_sample["tokens"]) == len(prepared_sample["labels"])

        return prepared_sample


def finetune_dataset(
    model_transform: Transform,
    *,
    source: str,
    message_transform_component: str,
    column_map: Optional[Dict[str, str]] = None,
    **load_dataset_kwargs,
) -> FinetuneDataset:

    return FinetuneDataset(
        source=source,
        message_transform=_get_component_from_path(message_transform_component)(),
        model_transform=model_transform,
        column_map=column_map,
        **load_dataset_kwargs,
    )
