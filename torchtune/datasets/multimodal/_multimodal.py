# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping

import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules.transforms import Transform


class MultimodalDataset(Dataset):
    """
    Class that supports combined image + text data with multiturn conversations.

    Use ``convert_to_messages`` to prepare your dataset into torchtune's Message
    structure::

        [
            # Images are contained in their own message
            Message.image_message(
                image=<image>,
            ),
            Message(
                role=<system|user|assistant>,
                content=<message>,
            ),
            ...
        ]

    For images, you must load in any file paths as ``PIL.Image.Image`` objects, then create
    individual messages for each image. There should not be any image tags in the
    user message text - use the image messages to replace them.

    This class supports multi-turn conversations. If a tokenizer sample with multiple
    turns does not fit within ``max_seq_len`` then it is truncated.

    Args:
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        model_transform (Transform): model-specific transform that takes in a sample dict and applies custom
            transforms on the keys. The tokenizer used by the model should be encapsulated in the model transform
            and should operate on the "messages" field. The keys returned by the model should be aligned with the
            expected inputs into the model.
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
        message_transform: Transform,
        model_transform: Transform,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._model_transform = model_transform
        self._data = load_dataset(source, **load_dataset_kwargs)
        self._message_transform = message_transform

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        transformed_sample = self._message_transform(sample)
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
