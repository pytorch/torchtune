# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from torchtune.data import format_content_with_images, load_image, Message
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms import Transform


class LlavaInstructToMessages(Transform):
    """
    Construct messages from a sample formatted similarly to
    `LLaVA-Instruct-150K <https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K>`_.

    Chat samples in the "conversations" column follow the ShareGPT format::

        {
            "image": "image0001.png",
            "conversations": [
                {
                    "from": "system" | "human" | "gpt",
                    "value": "<image> This is a sample image.",
                },
                ...
            ]
        }

    Image locations are indicated by "<image>" placeholder tags in the text content of each message.
    These are replaced by dictionaries indicating to the tokenizer where to place image tokens.
    Altogether, the above format is converted to torchtune's Message format::

        [
            {
                "role": "system" | "user" | "assistant",
                "content":
                    [
                        {"type": "image", "content": <PIL.Image.Image>},
                        {"type": "text", "content": "This is a sample image."},
                    ],
            },
            ...
        ]

    Args:
        train_on_input (bool): whether the prompt should remain unmasked. Default: False
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns ("conversations", "image")
            to the new column names in the dataset. Keys should be "conversations" and "image" and values should
            be the new column names. If None, keep the default "conversations" and "image".
            Default is None.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.
        images_dir (Optional[Path]): path to the directory containing the images. User is expected to download the COCO dataset.

    Raises:
        ValueError: If ``column_map`` is provided and ``conversations`` not in ``column_map``.
    """

    def __init__(
        self,
        train_on_input: bool = False,
        column_map: Optional[Dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
        images_dir: Optional[Path] = None,
    ):
        self.train_on_input = train_on_input
        self.new_system_prompt = new_system_prompt
        if column_map:
            if "image" not in column_map:
                raise ValueError(
                    f"Expected a key of 'image' in column_map but found {column_map.keys()}."
                )
            if "conversations" not in column_map:
                raise ValueError(
                    f"Expected a key of 'conversations' in column_map but found {column_map.keys()}."
                )
            self._column_map = column_map
        else:
            self._column_map = {"conversations": "conversations", "image": "image"}
        self.images_dir = images_dir

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        messages = []
        if self.new_system_prompt is not None:
            messages.append(
                Message(
                    role="system", content=self.new_system_prompt, masked=True, eot=True
                )
            )

        # Add in image stuffs / load from file
        for message in sample[self._column_map["conversations"]]:
            role = role_map[message["from"]]
            content = message["value"]
            if role == "system" and self.new_system_prompt is not None:
                continue
            if role == "user":
                image_path = sample[self._column_map["image"]]
                if self.images_dir is not None:
                    image_path = self.images_dir / image_path
                pil_image = load_image(image_path)
                content = format_content_with_images(
                    content,
                    image_tag="<image>",
                    images=[pil_image],
                )
            masked = (role != "assistant") and (not self.train_on_input)
            messages.append(Message(role=role, content=content, masked=masked))

        return {"messages": messages}


# TODO: point to Flamingo model transform as an example
def llava_instruct_dataset(
    model_transform: Transform,
    *,
    source: str = "liuhaotian/LLaVA-Instruct-150K",
    images_dir: str = "coco/",
    column_map: Optional[Dict[str, str]] = None,
    new_system_prompt: Optional[str] = None,
    train_on_input: bool = True,
    packed: bool = False,
    split: str = "train",
    data_files: str = "llava_instruct_150k.json",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    Support for family of image + text datasets similar to
    `LLaVA-Instruct-150K <https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K>`_
    from Hugging Face Datasets.

    To use this dataset, you must first download the COCO Train 2017 image dataset. You can do so
    by visiting https://cocodataset.org/#download or downloading it directly:

    .. code-block:: bash

        wget -c http://images.cocodataset.org/zips/train2017.zip
        unzip train2017.zip -d coco/

    The resulting directory should be passed into the model transform for loading
    and processing of the images.

    The model transform is expected to be a callable that applies pre-processing steps specific
    to a model. For multimodal datasets, this is expected to be at minimum a tokenizer and
    an image transform. The tokenizer will convert text sequences into token IDs after the dataset
    is converted to a list of :class:`~torchtune.data.Message`. The image transform will load the
    image and process it in accordance to the model's requirements.

    Here is a minimal example for illustrative purposes:

    .. code-block:: python

        from torchtune.models.llama3 import llama3_tokenizer
        from torchtune.models.clip import CLIPImageTransform
        from torchtune.modules.transforms import Transform

        class MyModelTransform(Transform):
            def __init__(
                self,
                tokenizer_path: str,
                max_seq_len: Optional[int] = None,
            ):
                self.tokenizer = llama3_tokenizer(tokenizer_path)
                self.image_transform = CLIPImageTransform()

            def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
                tokens, mask = self.tokenizer.tokenize_messages(sample["messages"])
                images = self.image_transform(sample["images"])
                return {
                    "tokens": tokens,
                    "mask": mask,
                    "images": images,
                }

    See :class:`~torchtune.datasets.SFTDataset` for more details about model transforms and
    message transforms.

    Args:
        model_transform (Transform): model-specific transform class that takes in a sample dict and applies custom
            transforms on the keys. It should consist of at minimum two components: text tokenization (called
            on the "messages" field) and image transform (called on the "images" field). The keys returned by
            the model transform should be aligned with the expected inputs into the model.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
        images_dir (str): path to the directory containing the images as you are expected to download the COCO dataset
            before using. Default is "coco/".
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns ("conversations")
            to the new column names in the dataset. If None, assume these are identical.
            Default is None.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        data_files (str): path to the json file to load as dataset. See the `dataset repo
            <https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main>`_ for options.
            Default is "llava_instruct_150k.json".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.

    Returns:
        Union[SFTDataset, PackedDataset]: dataset configured with source data and transform

    Raises:
        ValueError: If ``packed`` is True and ``max_seq_len`` is not set on the model_transform.

    Example:
        >>> llava_instruct_ds = llava_instruct_dataset(model_transform=model_transform)
        >>> for batch in Dataloader(llava_instruct_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    message_transform = LlavaInstructToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
        images_dir=Path(images_dir),
    )

    ds = SFTDataset(
        model_transform=model_transform,
        source=source,
        message_transform=message_transform,
        split=split,
        data_files=data_files,
        **load_dataset_kwargs,
    )
    if packed:
        if model_transform.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the model_transform."
            )
        return PackedDataset(ds, max_seq_len=model_transform.max_seq_len)
    return ds
