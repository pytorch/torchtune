# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import ChosenRejectedToMessages, CROSS_ENTROPY_IGNORE_IDX

from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform


class PreferenceDataset(Dataset):
    """
    Primary class for fine-tuning via preference modelling techniques (e.g. training
    a preference model for RLHF, or directly optimizing a model through DPO) on a
    preference dataset sourced from Hugging Face Hub, local files, or remote files. This
    class requires the dataset to have "chosen" and "rejected" model responses. These are
    typically either full conversations between user and assistant in separate columns::

        |  chosen                                |  rejected                              |
        |----------------------------------------|----------------------------------------|
        | [{"role": "user", "content": Q1},      | [{"role": "user", "content": Q1},      |
        |  {"role": "assistant", "content": A1}] |  {"role": "assistant", "content": A2}] |

    or a user prompt column with separate chosen and rejected assistant reponses::

        |  prompt  |  chosen  |  rejected  |
        |----------|----------|------------|
        |  Q1      |  A1      |  A2        |


    In the above case when the format is prompt-chosen-rejected, only single-turn interactions are supported.

    At a high level, this class will load the data from source and apply the following pre-processing steps when a
    sample is retrieved:

    1. Dataset-specific transform. This is typically unique to each dataset and extracts
       the necessary prompt and chosen/rejected columns into torchtune's :class:`~torchtune.data.Message`
       format, a standardized API for all model tokenizers.
    2. Tokenization with optional prompt template if configured


    All datasets are formatted into a list of :class:`~torchtune.data.Message`
    because preference datasets can be considered as chosen and rejected "conversations"
    with the model, or AI assistant. Thus, we can standardize all text content as messages
    in a conversation assigned to a role:

    - ``"user"`` messages contain the input prompt into the model
    - ``"assistant"`` messages are the response of the model and what you actually want
      to train for and compute loss directly against

    The :class:`~torchtune.data.Message` forms the core data unit that all tokenizer
    APIs expect. The key component of this class that ensures any dataset is transformed
    into this format is the ``message_transform``. This is a callable class that takes
    in a sample dictionary - typically a single row from the source dataset - that
    processes the sample in any configurable way to output a list of messages::

        [
            Message(
                role=<system|user|assistant|ipython>,
                content=<message>,
            ),
            ...
        ]

    For any custom dataset, use the ``message_transform`` to contain all pre-processing to
    return the list of messages.

    Args:
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        message_transform (Transform): callable that keys into the desired fields in the sample
            and converts text content to a list of :class:`~torchtune.data.Message`. It is expected that the final list
            of messages are stored in the ``"chosen"`` and ``"rejected"`` keys.
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
            Since PreferenceDataset only supports text data, it requires a
            :class:`~torchtune.modules.tokenizers.ModelTokenizer` instead of the ``model_transform`` in
            :class:`~torchtune.datasets.SFTDataset`.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.
    """

    def __init__(
        self,
        *,
        source: str,
        message_transform: Transform,
        tokenizer: ModelTokenizer,
        filter_fn: Optional[Callable] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
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

        # TODO: Truncation differs from original DPO repo
        # in DPO: first truncate prompts, then responses
        chosen_input_ids, chosen_masks = self._tokenizer.tokenize_messages(
            transformed_sample["chosen"],
        )
        chosen_labels = list(
            np.where(chosen_masks, CROSS_ENTROPY_IGNORE_IDX, chosen_input_ids)
        )

        rejected_input_ids, rejected_masks = self._tokenizer.tokenize_messages(
            transformed_sample["rejected"],
        )
        rejected_labels = list(
            np.where(rejected_masks, CROSS_ENTROPY_IGNORE_IDX, rejected_input_ids)
        )

        assert len(chosen_input_ids) == len(chosen_labels)
        assert len(rejected_input_ids) == len(rejected_labels)

        tokenized_dict = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            rejected_input_ids=rejected_input_ids,
            rejected_labels=rejected_labels,
        )

        return tokenized_dict


def preference_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> PreferenceDataset:
    """
    Configures a custom preference dataset comprising interactions between user and
    model assistant.

    This builder function can be used to configure a custom preference dataset directly from the yaml config
    as an alternative to :class:`~torchtune.datasets.PreferenceDataset`, as it is made to be config friendly.

    This function requires the dataset to have "chosen" and "rejected" columns. A single sample will share an
    identical system +/ user prompt between both "chosen" and "rejected" columns, followed by one or multiple
    turns of user and assistant messages::

        |  chosen                                |  rejected                              |
        |----------------------------------------|----------------------------------------|
        | [{"role": "user", "content": Q1},      | [{"role": "user", "content": Q1},      |
        |  {"role": "assistant", "content": C1}] |  {"role": "assistant", "content": R1}] |


    This example will be converted to:

    .. code-block:: python

        chosen_messages = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="C1"),
        ]

        rejected_messages = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="R1"),
        ]


    These lists of messages are then tokenized for model training. Currently, this function only supports
    conversations identical to :class:`~torchtune.data.OpenAIToMessages`, and does not support custom
    message formats.

    If your dataset does not follow this format, we recommend creating a custom message transform similar to
    :class:`~torchtune.data.ChosenRejectedToMessages` and using it in a custom dataset builder function similar
    to :class:`~torchtune.datasets.preference_dataset`.

    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is:
    set to ``False`` by default.

    - If ``train_on_input`` is True, the prompt is used during training and
      contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100).

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text"), pass
            in the filepath in ``data_files``, and set ``split="train"``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns "chosen" and "rejected"
            in the message transform :class:`~torchtune.data.ChosenRejectedToMessages` to the new column names in
            the dataset. Keys should be "chosen" and "rejected" and values should be the actual column names.
            If None, keep the default columns "chosen" and "rejected".
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        new_system_prompt (Optional[str]): if specified, prepend a system message to every sample for both chosen
            and rejected. This can serve as instructions to guide the model response. Setting this will OVERRIDE
            any system messages already present in the dataset. Default is None.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Examples:

    ::

        my_preference_dataset.json
        [
            {
                "chosen_conversations": [
                    {
                        "content": "What do I do when I have a hole in my trousers?",
                        "role": "user"
                    },
                    { "content": "Fix the hole.", "role": "assistant" }
                ],
                "rejected_conversations": [
                    {
                        "content": "What do I do when I have a hole in my trousers?",
                        "role": "user"
                    },
                    { "content": "Take them off.", "role": "assistant" }
                ]
            }
        ]

    ::

        >>> from torchtune.datasets import preference_dataset
        >>> column_map = {
        ...     "chosen": "chosen_conversations",
        ...     "rejected": "rejected_conversations"
        >>> }
        >>> dataset = preference_dataset(
        ...     tokenizer=tokenizer,
        ...     source="json",
        ...     column_map=column_map,
        ...     data_files="my_preference_dataset.json",
        ...     train_on_input=False,
        ...     split="train",
        >>> )
        >>> tokenizer.decode(dataset[0]["chosen_input_ids"], skip_special_tokens=True)
        What do I do when I have a hole in my trousers?Fix the hole.
        >>> tokenizer.decode(dataset[0]["rejected_input_ids"], skip_special_tokens=True)
        What do I do when I have a hole in my trousers?Take them off.

    This can also be accomplished via the yaml config:

    .. code-block:: yaml

        dataset:
          _component_: torchtune.datasets.preference_dataset
          source: json
          data_files: my_preference_dataset.json
          column_map:
            chosen: chosen_conversations
            rejected: rejected_conversations
          train_on_input: False
          split: train


    Returns:
        PreferenceDataset: The preference dataset built from source paired data.
    """

    message_transform = ChosenRejectedToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )

    return PreferenceDataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )
