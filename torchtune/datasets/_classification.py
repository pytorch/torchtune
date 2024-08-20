# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Mapping, Optional

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data import PromptTemplate

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

    1. Dataset-specific transform (Optional). This is typically unique to each dataset and extracts
       the the text column torchtune's :class:`~torchtune.data.Message`
       format, a standardized API for all model tokenizers. This step is not necessary for
       unstructured text.
    2. Tokenization


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
        model_transform: Transform,
        message_transform: Optional[Transform] = None,
        label_transform: Optional[Transform] = None,
        column_map: Dict[str, str] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:

        self._message_transform = message_transform
        self._model_transform = model_transform
        self._label_transform = label_transform

        self._column_map = column_map or {
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
            tokens = self._model_transform.encode(text=prompt, add_bos=True, add_eos=True)

        if self._label_transform is not None:
            label = self._label_transform(label)
        else:
            label = int(label)
        return {"tokens": tokens, "labels": label}
