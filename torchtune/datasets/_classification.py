# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, Optional

from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.modules.transforms import Transform


class ClassificationDataset(Dataset):
    """
    Primary class for fine-tuning classification models using a classification dataset sourced from
    Hugging Face Hub, local files, or remote files. This class requires the dataset to have
    "text" and "label" columns, which can be mapped to using the ``column_map`` arg.
    Typically the "text" column may be unstructured or structured text, and the "label" column
    is an integer::

        |  text                                  |  label                                 |
        |----------------------------------------|----------------------------------------|
        | [{"role": "user", "content": Q1},      | 1                                      |
        |  {"role": "assistant", "content": A1}] |                                        |


    In cases where ``label`` is not integer-encoded by default, you may use a ``label_transform``
    to map from the label column to an integer.

    At a high level, this class will load the data from source and apply the following pre-processing steps
    when a sample is retrieved:

    1. Dataset-specific text transform (Optional). This is typically unique to each dataset and extracts
       the the text column torchtune's :class:`~torchtune.data.Message`
       format, a standardized API for all model tokenizers.
    2. Dataset-specific label transform (Optional). This is also unique to each dataset, and is responsible
       for converting the "label" column into an integer target for classification. This could
       be as simple as a mapping e.g. ``{"label_0": 0, "label_1": 1...}``, but may also be used
       for non-trivial label conversion logic.
    2. Tokenization of the text column.

    Similar to :class:`~torchtune.datasets.PreferenceDataset`, classification datasets can be considered
    as either unstructured text, or arbitrary "conversations" between a user and model, or assistant.
    See :class:`~torchtune.datasets.PreferenceDataset` for more insight into how conversations are
    standardized as messages.

    Args:
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        model_transform (Transform): callable that applies model-specific pre-processing to the sample after the list of
            messages is created from ``message_transform``. This includes tokenization and any modality-specific
            transforms. It is expected to return at minimum ``"tokens"`` and ``"mask"`` keys.
        message_transform (Optional[Transform]): callable that keys into the desired fields in the sample
            and converts text content to a list of :class:`~torchtune.data.Message`. It is expected that the final list
            of messages are stored in the ``"messages"`` key.
        label_transform (Optional[Transform]): callable that converts content in a label column to a classification target.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "text" and "label" column names
            to the actual column names in the dataset. Default is None and set to "text" and "label".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.

    Raises:
        ValueError: If ``column_map`` is provided and ``text`` not in ``column_map``, or
            ``label`` not in ``column_map``.
    """

    def __init__(
        self,
        *,
        source: str,
        model_transform: Transform,
        message_transform: Optional[Transform] = None,
        label_transform: Optional[Transform] = None,
        column_map: Optional[Dict[str, str]] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:

        self._message_transform = message_transform
        self._model_transform = model_transform
        self._label_transform = label_transform

        if column_map:
            if "text" not in column_map:
                raise ValueError(
                    f"Expected a key of 'text' in column_map but found {column_map.keys()}."
                )
            if "label" not in column_map:
                raise ValueError(
                    f"Expected a key of 'label' in column_map but found {column_map.keys()}."
                )
            self._column_map = column_map
        else:
            self._column_map = {
                "text": "text",
                "label": "label",
            }
        self._data = load_dataset(source, **load_dataset_kwargs)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        prompt = sample[self._column_map["text"]]
        label = sample[self._column_map["label"]]
        if self._message_transform is not None:
            transformed_sample = self._message_transform({"messages": prompt})
            tokens = self._model_transform(transformed_sample)["tokens"]
        else:
            tokens = self._model_transform.encode(
                text=prompt, add_bos=True, add_eos=True
            )

        if self._label_transform is not None:
            label = self._label_transform(label)
        else:
            label = int(label)
        return {"tokens": tokens, "labels": label}
