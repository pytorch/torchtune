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
    def __init__(
        self,
        *,
        source: str,
        model_transform: Transform,
        message_transform: Optional[Transform] = None,
        label_transform: Optional[Transform] = None,
        column_map: Dict[str, str] = None,
        prompt_template: Optional[PromptTemplate] = None,
        filter_fn: Optional[Callable] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:

        self._message_transform = message_transform
        self._prompt_template = prompt_template
        self._model_transform = model_transform
        self._label_transform = label_transform

        self._column_map = column_map or {
            "text": "text",
            "label": "label",
        }
        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

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
            if self._prompt_template is not None:
                transformed_sample["messages"] = self._prompt_template(
                    transformed_sample["messages"]
                )
            tokens = self._model_transform(transformed_sample)["tokens"][1:-1]
        else:
            tokens = self._model_transform.encode(
                text=prompt, add_bos=False, add_eos=False
            )

        if self._label_transform is not None:
            label = self._label_transform(label)
        else:
            label = int(label)
        return {"tokens": tokens, "labels": label}
