# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data import (
    ChatFormat,
    CROSS_ENTROPY_IGNORE_IDX,
    JSONToMessages,
    Message,
    ShareGPTToMessages,
    validate_messages,
)
from torchtune.data._utils import deprecated
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.tokenizers import ModelTokenizer


@deprecated(msg="Please use `torchtune.datasets.SFTDataset` for custom chat data.")
class ChatDataset(Dataset):
    """
    Note:
        This class is deprecated and will be removed in a future release. Please use
        :class:`~torchtune.datasets.SFTDataset` or :func:`~torchtune.datasets.chat_dataset`
        for custom chat data.

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
    conversation_column: str,
    conversation_style: str,
    train_on_input: bool = False,
    packed: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    Configure a custom dataset with conversations between user and model assistant.

    This builder function can be used to configure a custom chat dataset directly from the yaml config
    as an alternative to :class:`~torchtune.datasets.SFTDataset`, as it is made to be config friendly.

    The dataset is expected to contain a single column with the conversations:

    .. code-block:: text

        |  conversations                         |
        |----------------------------------------|
        | [{"role": "user", "content": Q1},      |
        |  {"role": "assistant", "content": A1}] |

    This will be converted to:

    .. code-block:: python

        messages = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]

    This list of messages is then tokenized for model training.

    You may have a different structure for your conversations, such as different role names or
    different keys in the json structure. You can use the ``conversation_style`` parameter
    to choose from standard formats such as "sharegpt" (see :class:`~torchtune.data.ShareGPTToMessages`)
    or "json" (see :class:`~torchtune.data.JSONToMessages`). If your dataset is not in one of these
    formats, we recommend creating a custom message transform and using it in a custom dataset
    builder function similar to :class:`~torchtune.datasets.chat_dataset`.

    If your column names are different, use the ``conversation_column`` parameter to point
    towards the column with the conversations.

    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is
    set to ``False`` by default
    - If ``train_on_input`` is True, the prompt is used during training and
    contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text"), pass
            in the filepath in ``data_files``, and set ``split="train"``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        conversation_column (str): name of column containing the conversations.
        conversation_style (str): string specifying expected style of conversations in the dataset
            for automatic conversion to the :class:`~torchtune.data.Message` structure.
            Supported styles are: "sharegpt", "json"
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``,
            such as ``data_files`` or ``split``.

    Examples:

    ::

        my_dataset.json
        [
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": "What time is it in London?",
                    },
                    {
                        "from": "gpt",
                        "value": "It is 10:00 AM in London.",
                    },
                ],
            },
            {
                "conversations": [
                    ...
                ],
            },
            ...,
        ]

    ::

        >>> from torchtune.datasets import chat_dataset
        >>> dataset = chat_dataset(
        ...     tokenizer=tokenizer,
        ...     source="json",
        ...     data_files="my_dataset.json",
        ...     conversation_column="conversations",
        ...     conversation_style="sharegpt",
        ...     train_on_input=False,
        ...     packed=False,
        ...     split="train",
        ... )
        >>> tokens = dataset[0]["tokens"]
        >>> tokenizer.decode(tokens)
        "What time is it in London?It is 10:00 AM in London."

    This can also be accomplished via the yaml config:

    .. code-block:: yaml

        dataset:
          _component_: torchtune.datasets.chat_dataset
          source: json
          data_files: my_dataset.json
          conversation_column: conversations
          conversation_style: sharegpt
          train_on_input: False
          packed: False
          split: train

    Returns:
        Union[SFTDataset, PackedDataset]: the configured :class:`~torchtune.datasets.SFTDataset`
            or :class:`~torchtune.datasets.PackedDataset` if ``packed=True``

    Raises:
        ValueError: if the conversation format is not supported
    """
    if conversation_style == "sharegpt":
        message_transform = ShareGPTToMessages(
            train_on_input=train_on_input,
            column_map={"conversations": conversation_column},
        )
    elif conversation_style == "json":
        message_transform = JSONToMessages(
            train_on_input=train_on_input, column_map={"messages": conversation_column}
        )
    else:
        raise ValueError(f"Unsupported conversation style: {conversation_style}")

    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds
