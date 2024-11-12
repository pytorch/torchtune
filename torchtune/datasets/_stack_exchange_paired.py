# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Mapping, Optional

from torchtune.data import Message
from torchtune.datasets._preference import PreferenceDataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform


class StackExchangePairedToMessages(Transform):
    """
    Transform for converting datasets similar to the format in `Stack Exchange Paired dataset
    <https://huggingface.co/datasets/lvwerra/stack-exchange-paired>`_::

        |  prompt  |  chosen  |  rejected  |
        |----------|----------|------------|
        |  Q1      |  A1      |  A2        |

    into a list of chosen and rejected messages:

    .. code-block:: python

        chosen = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]
        rejected = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A2"),
        ]

    Args:
        train_on_input (bool): Whether the model is trained on the user prompt or not.
            Default is False.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "prompt",
            "chosen", and "rejected" column names to the actual column names in the dataset.
            Keys should be "prompt", "chosen", and "rejected" and values should be the actual column names.
            Default is None, keeping the default column names.
    """

    def __init__(
        self, train_on_input: bool = False, column_map: Optional[Dict[str, str]] = None
    ):
        self.train_on_input = train_on_input
        self._column_map = column_map

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        column_map = self._column_map or {}
        key_prompt = column_map.get("prompt", "prompt")
        key_chosen = column_map.get("chosen", "chosen")
        key_rejected = column_map.get("rejected", "rejected")

        chosen_messages = [
            Message(
                role="user", content=sample[key_prompt], masked=not self.train_on_input
            ),
            Message(role="assistant", content=sample[key_chosen]),
        ]

        rejected_messages = [
            Message(
                role="user", content=sample[key_prompt], masked=not self.train_on_input
            ),
            Message(role="assistant", content=sample[key_rejected]),
        ]

        return {"chosen": chosen_messages, "rejected": rejected_messages}


def stack_exchange_paired_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "lvwerra/stack-exchange-paired",
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> PreferenceDataset:
    """
    Family of preference datasets similar to the `Stack Exchange Paired dataset
    <https://huggingface.co/datasets/lvwerra/stack-exchange-paired>`_.

    It is recommended to configure the tokenizer with the :class:`~torchtune.data.QuestionAnswerTemplate`
    in conjunction with this dataset.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details. Default is ``lvwerra/stack-exchange-paired``.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "prompt",
            "chosen", and "rejected" column names to the actual column names in the dataset.
            Keys should be "prompt", "chosen", and "rejected" and values should be the actual column names.
            Default is None, keeping the default column names.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Returns:
        PreferenceDataset: The preference dataset built from source paired data.
    """

    column_map = column_map or {
        "prompt": "question",
        "chosen": "response_j",
        "rejected": "response_k",
    }

    message_transform = StackExchangePairedToMessages(
        train_on_input=train_on_input, column_map=column_map
    )

    return PreferenceDataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        split=split,
        data_dir="data/rl",
        **load_dataset_kwargs,
    )
