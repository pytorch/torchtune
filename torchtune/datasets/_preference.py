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
    InstructTemplate,
    Message,
    validate_messages,
)

from torchtune.modules.tokenizers import ModelTokenizer


class InstructPreferenceDataset(Dataset):
    """
    Class that supports any custom preference dataset with instruction-based prompts and a
    configurable template.

    The general flow from loading a sample to tokenized prompt is:
    load sample -> apply transform -> format into template -> tokenize

    If the column/key names differ from the expected names in the :class:`~torchtune.data.InstructTemplate`,
    then the ``column_map`` argument can be used to provide this mapping.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        template (InstructTemplate): template used to format the prompt. If the placeholder variable
            names in the template do not match the column/key names in the dataset, use ``column_map`` to map them.
        transform (Optional[Callable]): transform to apply to the sample before formatting to the template.
            Default is None.
        column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names in the template
            to the column/key names in the sample. The expected column names are "prompt", "chosen", and "rejected".
            The default mapping uses identical keys i.e. ``{"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"}``.
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
        self._column_map = (
            column_map
            if column_map is not None
            else {
                "prompt": "prompt",
                "chosen": "chosen",
                "rejected": "rejected",
            }
        )
        self.max_seq_len = max_seq_len

        # TODO move this to the DPO Recipe / to a DPO specific transform
        if max_seq_len is not None:
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

        chosen_message = [
            Message(role="user", content=prompt, masked=True),
            Message(
                role="assistant", content=transformed_sample[self._column_map["chosen"]]
            ),
        ]

        rejected_message = [
            Message(role="user", content=prompt, masked=True),
            Message(
                role="assistant",
                content=transformed_sample[self._column_map["rejected"]],
            ),
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


class ChatPreferenceDataset(Dataset):
    """
    Class that supports any custom preference dataset with chat-based prompts and a
    configurable template. See :class:`~torchtune.data.ChatDataset` for more details.

    The general flow from loading a sample to tokenized prompt is:
    load sample -> apply transform -> format into template -> tokenize

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
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
        column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names in the template
            to the column/key names in the sample. The expected column names are "chosen", and "rejected".
            The default mapping uses identical keys i.e. ``{"chosen": "chosen", "rejected": "rejected"}``.
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        source: str,
        convert_to_messages: Callable[[Mapping[str, Any]], List[Message]],
        chat_format: Optional[ChatFormat] = None,
        column_map: Optional[Dict[str, str]] = None,
        max_seq_len: Optional[int] = None,
        train_on_input: bool = False,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:

        if not isinstance(chat_format(), ChatFormat):
            raise ValueError(
                f"chat_format must be a ChatFormat class, not {type(chat_format())}"
            )
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self._convert_to_messages = convert_to_messages
        self.chat_format = chat_format

        self._column_map = (
            column_map
            if column_map is not None
            else {
                "chosen": "chosen",
                "rejected": "rejected",
            }
        )

        self.train_on_input = train_on_input
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:

        # this should support both sharegpt and openai formats
        chosen_messages = self._convert_to_messages(
            {"conversations": sample[self._column_map["chosen"]]}, self.train_on_input
        )
        rejected_messages = self._convert_to_messages(
            {"conversations": sample[self._column_map["rejected"]]}, self.train_on_input
        )

        if self.chat_format is not None:
            chosen_messages = self.chat_format.format(chosen_messages)
            rejected_messages = self.chat_format.format(rejected_messages)

        validate_messages(chosen_messages)
        validate_messages(rejected_messages)

        chosen_input_ids, c_masks = self._tokenizer.tokenize_messages(
            chosen_messages, self.max_seq_len
        )
        chosen_labels = list(
            np.where(c_masks, CROSS_ENTROPY_IGNORE_IDX, chosen_input_ids)
        )

        rejected_input_ids, r_masks = self._tokenizer.tokenize_messages(
            rejected_messages, self.max_seq_len
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


def chat_preference_dataset(
    *,
    tokenizer: ModelTokenizer,
    source: str,
    conversation_style: str,
    chat_format: Optional[str] = None,
    max_seq_len: int,
    train_on_input: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> ChatPreferenceDataset:
    """
    Build a configurable preference dataset with conversations. This method should be
    used to configure a custom chat-format preference dataset from the yaml config instead of
    using :class:`~torchtune.datasets.ChatPreferenceDataset` directly, as it is made to be config friendly.
    An example of a dataset which this config would support is
        https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned
    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        conversation_style (str): string specifying expected style of conversations in the dataset
            for automatic conversion to the :class:`~torchtune.data.Message` structure. Supported styles are: "sharegpt", "openai"
        chat_format (Optional[str]): full import path of :class:`~torchtune.data.ChatFormat` class used to format the messages.
            See the description in :class:`~torchtune.datasets.ChatDataset` for more details. For a list of all
            possible chat formats, check out :ref:`chat_formats`. Default: None.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Examples:
        >>> from torchtune.datasets import chat_preference_dataset
        >>> dataset = chat_preference_dataset(
        ...   tokenizer=tokenizer,
        ...   source="mlabonne/orpo-dpo-mix-40k",
        ...   conversation_style="openai",
        ...   chat_format="torchtune.data.ChatMLFormat",
        ...   max_seq_len=2096,
        ...   train_on_input=False
        ... )

    This can also be accomplished via the yaml config::

        dataset:
            _component_: torchtune.datasets.chat_preference_dataset
            source: mlabonne/orpo-dpo-mix-40k
            conversation_style: openai
            chat_format: torchtune.data.ChatMLFormat
            max_seq_len: 2096
            train_on_input: False

    Returns:
        ChatPreferenceDataset: the configured :class:`~torchtune.datasets.ChatPreferenceDataset`.

    Raises:
        ValueError: if the conversation format is not supported
    """
    if conversation_style == "sharegpt":
        convert_to_messages = get_sharegpt_messages
    elif conversation_style == "openai":
        convert_to_messages = get_openai_messages
    else:
        raise ValueError(f"Unsupported conversation style: {conversation_style}")

    return ChatPreferenceDataset(
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
