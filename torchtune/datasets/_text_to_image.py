# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data._utils import load_image
from torchtune.modules.transforms import Transform


class TextToImageDataset(Dataset):
    """
    Dataset of image-text pairs from HuggingFace or local files.

    Args:
        source (str): Either a path to a huggingface dataset or the type of the local dataset (e.g. json)
            See https://huggingface.co/docs/datasets/en/loading for more info
        image_dir (Optional[str]): The directory that image paths are relative to.
            Default: None (the image paths are absolute paths)
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "text"/"image" (and optionally "id")
            column names to the actual column names in the dataset. Keys should be "text" and "image", and values
            should be the actual column names.
            Default: None (use default column names)
        include_id (bool): Set to True if the dataset includes unique ids and you'd like to include them in the output
            (e.g. for preprocessing purposes).
            Default: False
        model_transform (Transform): Callable that applies model-specific preprocessing to the sample.
            See :class:`~torchtune.models.flux.FluxTransform` for an example.
        filter_fn (Optional[Callable]): Callable used to filter the dataset prior to any pre-processing.
            See https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter
            Default: None
        **load_dataset_kwargs (Dict[str, Any]): Additional keyword arguments to pass to `load_dataset`.
            See https://huggingface.co/docs/datasets/en/loading

    Raises:
        ValueError: if `column_map` is invalid.
    """

    def __init__(
        self,
        *,
        source: str,
        image_dir: Optional[str] = None,
        column_map: Optional[Dict[str, str]] = None,
        include_id: bool = False,
        model_transform: Transform,
        filter_fn: Optional[Callable] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._image_dir = Path(image_dir) if image_dir is not None else None

        if column_map is None:
            self._column_map = {"text": "text", "image": "image"}
            if include_id:
                self._column_map["id"] = "id"
        else:
            if "text" not in column_map:
                raise ValueError(
                    f"Expected a key of 'text' in column_map but found {column_map.keys()}."
                )
            if "image" not in column_map:
                raise ValueError(
                    f"Expected a key of 'image' in column_map but found {column_map.keys()}."
                )
            if include_id and "id" not in column_map:
                raise ValueError(
                    f"Expected a key of 'id' in column_map but found {column_map.keys()}."
                )
            self._column_map = column_map

        self._model_transform = model_transform

        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

        self._include_id = include_id

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._data[index]

        id = sample[self._column_map["id"]]
        text = sample[self._column_map["text"]]
        image_path = sample[self._column_map["image"]]

        image = load_image(
            Path(image_path)
            if self._image_dir is None
            else self._image_dir / image_path
        )

        transformed_sample = self._model_transform({"text": text, "image": image})

        if self._include_id:
            transformed_sample["id"] = id
        return transformed_sample


def text_to_image_dataset(
    model_transform: Transform,
    *,
    source: str,
    image_dir: Optional[str] = None,
    column_map: Optional[Dict[str, str]] = None,
    include_id: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> TextToImageDataset:
    """
    Builder for a dataset of image-text pairs from HuggingFace or local files.

    ---

    For a local dataset in the following format:

    /path/to/my_image_folder
        - 0.png
        ...

    /path/to/my_dataset.json
    [
        {"text": "my caption", "image": "0.png"},
        ...
    ]

    Your dataset config will look like this:

    .. code-block:: yaml
        dataset:
          _component_: torchtune.datasets.text_to_image_dataset
          source: json
          data_files: /path/to/my_dataset.json
          image_dir: /path/to/my_image_folder

    ---

    Args:
        model_transform (Transform): Callable that applies model-specific preprocessing to the sample.
            See :class:`~torchtune.models.flux.FluxTransform` for an example.
        source (str): Either a path to a huggingface dataset or the type of the local dataset (e.g. json)
            See https://huggingface.co/docs/datasets/en/loading for more info
        image_dir (Optional[str]): The directory that image paths are relative to.
            Default: None (the image paths are absolute paths)
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "text"/"image" (and optionally "id")
            column names to the actual column names in the dataset. Keys should be "text" and "image", and values
            should be the actual column names.
            Default: None (use default column names)
        include_id (bool): Set to True if the dataset includes unique ids and you'd like to include them in the output
            (e.g. for preprocessing purposes).
            Default: False
        filter_fn (Optional[Callable]): Callable used to filter the dataset prior to any pre-processing.
            See https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter
            Default: None
        split (str): Load a specific subset of a dataset.
            See https://huggingface.co/docs/datasets/en/loading
            Default: "train"
        **load_dataset_kwargs (Dict[str, Any]): Additional keyword arguments to pass to `load_dataset`.
            See https://huggingface.co/docs/datasets/en/loading

    Returns:
        TextToImageDataset
    """
    ds = TextToImageDataset(
        source=source,
        image_dir=image_dir,
        column_map=column_map,
        include_id=include_id,
        model_transform=model_transform,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )
    return ds
