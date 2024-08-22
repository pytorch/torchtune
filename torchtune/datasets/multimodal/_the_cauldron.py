# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, Optional, Union

from torchtune.data import Message
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms import Transform


class TheCauldronToMessages(Transform):
    """
    Construct messages from a sample formatted similarly to
    `The Cauldron dataset<https://huggingface.co/datasets/HuggingFaceM4/the_cauldron>_`.

    Image placeholders are prepended to the text in the ``Message`` content. Images in the
    dataset are expected to be a list of a single PIL image, so they are simply passed through
    with an optional column remapping if ``column_map`` is specified.

    Args:
        train_on_input (bool): Whether the model is trained on the user prompt or not.
            Default is True.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "images"
            and "texts" column names to the actual column names in the dataset. Default is None,
            keeping the default column names.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Default is None.

    Raises:
        ValueError: If ``column_map`` is provided and ``images`` not in ``column_map``, or
            ``texts`` not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: bool = True,
        column_map: Optional[Dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
    ):
        self.train_on_input = train_on_input
        self.new_system_prompt = new_system_prompt
        if column_map is not None:
            if "images" not in column_map:
                raise ValueError(
                    "column_map must contain 'images' as a key if specified"
                )
            if "texts" not in column_map:
                raise ValueError(
                    "column_map must contain 'texts' as a key if specified"
                )
            self._column_map = column_map
        else:
            self._column_map = {"images": "images", "texts": "texts"}

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        messages = []
        for message in sample[self._column_map["texts"]]:
            messages.append(
                Message(
                    role="user",
                    content=[
                        {"type": "image"},
                        {"type": "text", "content": message["user"]},
                    ],
                    masked=not self.train_on_input,
                )
            )
            messages.append(
                Message(
                    role="assistant",
                    content=[{"type": "text", "content": message["assistant"]}],
                )
            )

        if self.new_system_prompt is not None:
            messages = [
                Message(
                    role="system", content=self.new_system_prompt, masked=True, eot=True
                )
            ] + messages

        return {"messages": messages, "images": sample[self._column_map["images"]]}


def the_cauldron_dataset(
    model_transform: Transform,
    *,
    subset: str,
    source: str = "HuggingFaceM4/the_cauldron",
    column_map: Optional[Dict[str, str]] = None,
    new_system_prompt: Optional[str] = None,
    train_on_input: bool = True,
    packed: bool = False,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    Support for family of image + text datasets similar to
    `The Cauldron<https://huggingface.co/datasets/HuggingFaceM4/the_cauldron>_`
    from Hugging Face Datasets.

    Args:
        model_transform (Transform): model-specific transform that takes in a sample dict and applies custom
            transforms on the keys. The tokenizer used by the model should be encapsulated in the model transform
            and should operate on the "messages" field. Any model-specific image transforms should operate on
            the "images" field. The keys returned by the model should be aligned with the
            expected inputs into the model.
        subset (str): name of the subset of the dataset to load. See the `dataset card
            <https://huggingface.co/datasets/HuggingFaceM4/the_cauldron>`_ for options.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details. Default is ``HuggingFaceM4/the_cauldron``.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "images"
            and "texts" column names to the actual column names in the dataset. Default is None,
            keeping the default column names.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.

    Returns:
        Union[SFTDataset, PackedDataset]: dataset configured with source data and transform

    Raises:
        ValueError: If ``packed`` is True and ``max_seq_len`` is not set on the model_transform.

    Example:
        >>> cauldron_ds = the_cauldron_dataset(model_transform=model_transform, subset="ai2d")
        >>> for batch in Dataloader(cauldron_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    message_transform = TheCauldronToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
    )

    ds = SFTDataset(
        model_transform=model_transform,
        source=source,
        message_transform=message_transform,
        name=subset,
        split=split,
        **load_dataset_kwargs,
    )
    if packed:
        if model_transform.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the model_transform."
            )
        return PackedDataset(ds, max_seq_len=model_transform.max_seq_len)
    return ds
