# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Mapping

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX

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

    At a high level, this class will load the data from source and apply the following pre-processing steps
    when a sample is retrieved:

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
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._message_transform = message_transform
        self._data = load_dataset(source, **load_dataset_kwargs)

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
