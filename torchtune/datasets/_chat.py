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
from torchtune.data import (
    ChatFormat,
    CROSS_ENTROPY_IGNORE_IDX,
    get_openai_messages,
    get_sharegpt_messages,
    Message,
    validate_messages,
)
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.tokenizers import ModelTokenizer


class ChatDataset(Dataset):
    """
    Class that supports any custom dataset with multiturn conversations.

    The general flow from loading a sample to tokenized prompt is:
    load sample -> apply transform -> foreach turn{format into template -> tokenize}

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
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        convert_to_messages (Callable[[Mapping[str, Any]], List[Message]]): function that keys into the desired field in the sample
            and converts to a list of :class:`~torchtune.data.Message` that follows the Llama format with the expected keys
        chat_format (Optional[ChatFormat]): template used to format the chat. This is used to add structured text around the actual
            messages, such as the [INST] tags in Llama2 and in Mistral. The extra text will still get tokenized as normal text, not
            as special tokens. In models like Llama3 where the tokenizer adds tags as special tokens, ``chat_format`` is not needed,
            unless you want to structure messages in a particular way for inference.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``,
            such as ``data_files`` or ``split``.

    Raises:
        ValueError: if ``chat_format`` is not an instance of :class:`torchtune.data.ChatFormat`.
    """

    def __init__(
        self,
        *,
        tokenizer: ModelTokenizer,
        source: str,
        convert_to_messages: Callable[[Mapping[str, Any]], List[Message]],
        chat_format: Optional[ChatFormat] = None,
        max_seq_len: int,
        train_on_input: bool = False,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        if chat_format is not None and not isinstance(chat_format(), ChatFormat):
            raise ValueError(
                f"chat_format must be a ChatFormat class, not {type(chat_format())}"
            )

        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self._convert_to_messages = convert_to_messages
        self.chat_format = chat_format
        self.max_seq_len = max_seq_len
        self.train_on_input = train_on_input

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        messages = self._convert_to_messages(sample, self.train_on_input)
        if self.chat_format is not None:
            messages = self.chat_format.format(messages)
        validate_messages(messages)
        tokens, mask = self._tokenizer.tokenize_messages(
            messages,
        )
        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, tokens))
        assert len(tokens) == len(labels)

        return {"tokens": tokens, "labels": labels}


def chat_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    conversation_style: str,
    chat_format: Optional[str] = None,
    max_seq_len: int,
    train_on_input: bool = False,
    packed: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> ChatDataset:
    """
    Build a configurable dataset with conversations. This method should be
    used to configure a custom chat dataset from the yaml config instead of
    using :class:`~torchtune.datasets.ChatDataset` directly, as it is made to be config friendly.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        conversation_style (str): string specifying expected style of conversations in the dataset
            for automatic conversion to the :class:`~torchtune.data.Message` structure. Supported styles are: "sharegpt", "openai"
        chat_format (Optional[str]): full import path of :class:`~torchtune.data.ChatFormat` class used to format the messages.
            See the description in :class:`~torchtune.datasets.ChatDataset` for more details. For a list of all
            possible chat formats, check out :ref:`chat_formats`. Default: None.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``,
            such as ``data_files`` or ``split``.

    Examples:
        >>> from torchtune.datasets import chat_dataset
        >>> dataset = chat_dataset(
        ...   tokenizer=tokenizer,
        ...   source="HuggingFaceH4/no_robots",
        ...   conversation_style="sharegpt",
        ...   chat_format="torchtune.data.ChatMLFormat",
        ...   max_seq_len=2096,
        ...   train_on_input=True
        ... )

    This can also be accomplished via the yaml config::

        dataset:
            _component_: torchtune.datasets.chat_dataset
            source: HuggingFaceH4/no_robots
            conversation_style: sharegpt
            chat_format: torchtune.data.ChatMLFormat
            max_seq_len: 2096
            train_on_input: True

    Returns:
        ChatDataset or PackedDataset: the configured :class:`~torchtune.datasets.ChatDataset`
            or :class:`~torchtune.datasets.PackedDataset` if ``packed=True``

    Raises:
        ValueError: if the conversation format is not supported
    """
    if conversation_style == "sharegpt":
        convert_to_messages = get_sharegpt_messages
    elif conversation_style == "openai":
        convert_to_messages = get_openai_messages
    else:
        raise ValueError(f"Unsupported conversation style: {conversation_style}")

    ds = ChatDataset(
        tokenizer=tokenizer,
        source=source,
        convert_to_messages=convert_to_messages,
        chat_format=_get_component_from_path(chat_format)
        if chat_format is not None
        else None,
        max_seq_len=max_seq_len,
        train_on_input=train_on_input,
        **load_dataset_kwargs,
    )
    return (
        PackedDataset(ds, max_seq_len=max_seq_len, padding_idx=tokenizer.pad_id)
        if packed
        else ds
    )
