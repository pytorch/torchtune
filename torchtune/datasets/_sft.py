# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Mapping, Optional, Dict

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._messages import validate_messages

from torchtune.modules.transforms import Transform
from torchtune.data._metrics import StandardMetricTransform
from torchtune.datasets._hf_iterable import HfIterableDataset


class SFTDataset(Dataset):
    """
    Primary class for creating any dataset for supervised fine-tuning either from
    Hugging Face Hub, local files, or remote files. This class supports instruct,
    chat, tool, or multimodal data for fine-tuning. At a high level, this class
    will load the data from source and apply the following pre-processing steps
    when a sample is retrieved:

    1. Dataset-specific transform. This is typically unique to each dataset and extracts
       the necessary columns into torchtune's :class:`~torchtune.data.Message` format,
       a standardized API for all model tokenizers.
    2. Model-specific transform or tokenization with optional prompt template


    All datasets are formatted into a list of :class:`~torchtune.data.Message`
    because for fine-tuning, datasets can be considered as "conversations" with the model,
    or AI assistant. Thus, we can standardize all text content as messages in a conversation assigned to
    a role:

    - ``"system"`` messages contain the system prompt
    - ``"user"`` messages contain the input prompt into the model
    - ``"assistant"`` messages are the response of the model and what you actually want
      to train for and compute loss directly against
    - ``"ipython"`` messages are the return from a tool call

    Chat datasets are multiple rounds of user-assistant messages. Instruct datasets
    are typically a single round involving a specific instruction and the model's response.
    Tool datasets are a type of chat dataset that includes ipython messages. Multimodal
    datasets are a type of chat dataset that incorporates media into the user messages.

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

    Any model-specific pre-processing that needs to happen can be configured with the ``model_transform``
    parameter. This is another callable class that contains any custom logic tied to the
    model you are fine-tuning and will carry over to inference. For example, text + image
    multimodal datasets requires processing the images in a way specific to the vision
    encoder being used by the model and is agnostic to the specific dataset.

    Tokenization is handled by the ``model_transform``. All
    :class:`~torchtune.modules.transforms.tokenizers.ModelTokenizer` can be treated as
    a ``model_transform`` since it uses the model-specific tokenizer to transform the
    list of messages outputted from the ``message_transform`` into tokens used by the
    model for training. Text-only datasets will simply pass the
    :class:`~torchtune.modules.transforms.tokenizers.ModelTokenizer` into ``model_transform``.
    Tokenizers handle prompt templating, if configured.

    Args:
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        message_transform (Transform): callable that keys into the desired fields in the sample
            and converts text content to a list of :class:`~torchtune.data.Message`. It is expected that the final list
            of messages are stored in the ``"messages"`` key. See :ref:`message_transform_usage_label` for details.
        model_transform (Transform): callable that applies model-specific pre-processing to the sample after the list of
            messages is created from ``message_transform``. This includes tokenization and any modality-specific
            transforms. It is expected to return at minimum ``"tokens"`` and ``"mask"`` keys.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        filter_kwargs (Optional[dict[str, Any]]): additional keyword arguments to pass to ``filter_fn``.
        **load_dataset_kwargs (dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.
    """

    def __init__(
        self,
        *,
        source: str,
        message_transform: Transform,
        model_transform: Transform,
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[dict[str, Any]] = None,
        **load_dataset_kwargs: dict[str, Any],
    ) -> None:
        self._message_transform = message_transform
        self._model_transform = model_transform

        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            if filter_kwargs is None:
                filter_kwargs = {}
            self._data = self._data.filter(filter_fn, **filter_kwargs)

        self._prepare_sample = SFTTransform(
            message_transform=self._message_transform,
            model_transform=self._model_transform,
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)


class SFTTransform(Transform):
    def __init__(
        self,
        message_transform: Optional[Transform] = None,
        model_transform: Optional[Transform] = None,
    ):
        if message_transform is None and model_transform is None:
            raise ValueError(
                "At least one of message_transform or model_transform must be provided."
            )
        self._message_transform = message_transform
        self._model_transform = model_transform

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        if self._message_transform is not None:
            transformed_sample = self._message_transform(sample)
            if "messages" in transformed_sample:
                validate_messages(transformed_sample["messages"])
        else:
            transformed_sample = sample

        if self._model_transform is not None:
            tokenized_dict = self._model_transform(transformed_sample)

            if not ("tokens" in tokenized_dict and "mask" in tokenized_dict):
                keys_str = ", ".join(tokenized_dict.keys())
                error_message = (
                    "model_transform returned the following keys: "
                    f"{keys_str}. Must return 'tokens' and 'mask' as keys."
                )
                raise ValueError(error_message)

            # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
            # Shift labels to be off by 1 from the logits.
            # Padding added at the end so we dont need to slice the logits.
            tokenized_dict["labels"] = list(
                np.where(
                    tokenized_dict["mask"][1:],
                    CROSS_ENTROPY_IGNORE_IDX,
                    tokenized_dict["tokens"][1:],
                )
            )
            tokenized_dict["labels"].append(CROSS_ENTROPY_IGNORE_IDX)
            assert len(tokenized_dict["tokens"]) == len(tokenized_dict["labels"])
        else:
            tokenized_dict = transformed_sample

        return tokenized_dict


class SFTOutputTransform(Transform):
    """
    Output transform to be used in SFT recipes as an input to TuneIterableDataset.
    It takes tokenized inputs with "tokens" and "mask" keys and
    creates the "labels" key for SFT training.
    
    The labels are created by:
    1. Shifting tokens by 1 position (for autoregressive training)
    2. Masking positions where mask[1:] is True with CROSS_ENTROPY_IGNORE_IDX
    3. Adding CROSS_ENTROPY_IGNORE_IDX at the end
    """
    
    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        # Create a copy to avoid modifying the original
        tokenized_dict = dict(sample)
        
        if not ("tokens" in tokenized_dict and "mask" in tokenized_dict):
            keys_str = ", ".join(tokenized_dict.keys())
            raise ValueError(
                f"SFTOutputTransform expects 'tokens' and 'mask' keys. "
                f"Got keys: {keys_str}"
            )
        
        # Create labels for SFT training
        tokenized_dict["labels"] = list(
            np.where(
                tokenized_dict["mask"][1:],
                CROSS_ENTROPY_IGNORE_IDX,
                tokenized_dict["tokens"][1:],
            )
        )
        tokenized_dict["labels"].append(CROSS_ENTROPY_IGNORE_IDX)
        assert len(tokenized_dict["tokens"]) == len(tokenized_dict["labels"])
        
        return tokenized_dict


def sft_iterable_dataset(
    model_transform: Transform, 
    *,
    message_transform: Transform,
    shuffle_buffer_size: Optional[int] = 1000,
    seed: int = 42,
    num_shards_per_rank: int = 64,
    dataset_name: Optional[str] = None,
    filter_fn: Optional[Callable] = None,
    filter_kwargs: Optional[Dict[str, Any]] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> HfIterableDataset:
    """
    Creates an SFT-ready iterable dataset with appropriate output transform.
    
    Args:
        model_transform (Transform): Usually the tokenizer
        message_transform (Transform): Transform to convert raw data to messages
        shuffle_buffer_size (Optional[int]): Buffer size for shuffling  
        seed (int): Random seed for shuffling
        num_shards_per_rank (int): Target shards per worker
        dataset_name (Optional[str]): Name for metrics namespacing
        filter_fn (Optional[Callable]): Filter function
        filter_kwargs (Optional[Dict[str, Any]]): Filter function kwargs
        **load_dataset_kwargs: Args passed to load_dataset
        
    Returns:
        HfIterableDataset: Configured for SFT training
        
    Example:
        >>> from torchtune.data import AlpacaToMessages
        >>> message_transform = AlpacaToMessages(train_on_input=False)
        >>> ds = sft_iterable_dataset(
        ...     message_transform=message_transform,
        ...     model_transform=tokenizer,
        ...     path="tatsu-lab/alpaca"
        ... )
    """

    output_transform = SFTOutputTransform()
    
    return HfIterableDataset(
        message_transform=message_transform,
        model_transform=model_transform,
        output_transform=output_transform,  
        metric_transform=StandardMetricTransform(),
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        num_shards_per_rank=num_shards_per_rank,
        dataset_name=dataset_name,
        filter_fn=filter_fn,
        filter_kwargs=filter_kwargs,
        **load_dataset_kwargs,
    )
