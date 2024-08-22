# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Union

from torchtune.data import ShareGPTToMessages
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms import Transform


def llava_instruct_dataset(
    model_transform: Transform,
    *,
    coco_image_dir: str,
    source: str = "liuhaotian/LLaVA-Instruct-150K",
    column_map: Optional[Dict[str, str]] = None,
    new_system_prompt: Optional[str] = None,
    image_tag: Optional[str] = "<image>",
    train_on_input: bool = True,
    packed: bool = False,
    split: str = "train",
    data_files: str = "llava_instruct_150k.json",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    Support for family of image + text datasets similar to
    `LLaVA-Instruct-150K <https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K>_`
    from Hugging Face Datasets.

    To use this dataset, you must first download the COCO Train 2017 image dataset. You can do so
    by visiting https://cocodataset.org/#download or downloading it directly:

    .. code-block:: bash

        wget -c http://images.cocodataset.org/zips/train2017.zip
        unzip train2017.zip -d coco/

    Then, you must pass in the directory containing all the images to the ``image_dir`` argument
    so each file can be loaded as PIL images in the transform. In the example above, you would
    set ``image_dir="coco/train2017"``.

    Args:
        model_transform (Transform): model-specific transform that takes in a sample dict and applies custom
            transforms on the keys. The tokenizer used by the model should be encapsulated in the model transform
            and should operate on the "messages" field. Any model-specific image transforms should operate on
            the "images" field. The keys returned by the model should be aligned with the
            expected inputs into the model.
        coco_image_dir (str): path to the directory containing the COCO Train 2017 images.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns ("conversations")
            to the new column names in the dataset. If None, assume these are identical.
            Default is None.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.
        image_tag (Optional[str]): if specified, split the raw text content by the specified ``image_tag``
            and use placeholders for where the images are present in the text for proper tokenization. Set
            this if your dataset contains images and uses a specific string (ex: "<image>") to indicate the
            presence of an image. Leave this as None if your dataset does not contain images. Default is "<image>".
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        data_files (str): path to the json file to load as dataset. See the `dataset repo
            <https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main>`_ for options.
            Default is "llava_instruct_150k.json".
        **load_dataset_kwargs: Additional keyword arguments to pass to ``load_dataset``.

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

    message_transform = ShareGPTToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
        image_tag=image_tag,
        image_dir=coco_image_dir,
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
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=model_transform.max_seq_len)
    return ds
