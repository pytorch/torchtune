# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Mapping, Optional

from torchtune.data._messages import Message
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms import Transform


class TheCauldronToMessages(Transform):
    """
    Construct messages from a sample formatted similarly to
    `The Cauldron dataset <https://huggingface.co/datasets/HuggingFaceM4/the_cauldron>`_.

    Image placeholders are prepended to the text in the ``Message`` content. Images in the
    dataset are expected to be a list of a single PIL image, so they are simply passed through
    to the model transform with an optional column remapping if ``column_map`` is specified.

    For example, a dataset row::

        {
            "texts": [
                {
                    "user": "What are in these images.",
                    "assistant": "They are images of dogs.",
                },
                ...
            ],
            "images": [
                [PIL.Image.Image, PIL.Image.Image],
            ],
        }

    will be converted to::

        [
            Message(
                role = "user",
                content = [
                    {"type": "image", "content": <PIL.Image.Image>},
                    {"type": "image", "content": <PIL.Image.Image>},
                    {"type": "text", "content": "What are in these images."},
                ],
            ),
            Message(
                role = "assistant",
                content = [
                    {"type": "text", "content": "They are images of dogs."},
                ],
            ),
            ...
        ]

    Args:
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "texts"
            column names to the actual column names in the dataset. Default is None,
            keeping the default column names.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Default is None.

    Raises:
        ValueError: If ``column_map`` is provided and ``texts`` not in ``column_map``.
    """

    def __init__(
        self,
        column_map: Optional[Dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
    ):
        self.new_system_prompt = new_system_prompt
        if column_map is not None:
            if "images" not in column_map:
                raise ValueError(
                    "column_map must map 'images' to your expected column name if specified"
                )
            if "texts" not in column_map:
                raise ValueError(
                    "column_map must map 'texts' to your expected column name if specified"
                )
            self._column_map = column_map
        else:
            self._column_map = {"texts": "texts", "images": "images"}

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        # Dataset images to be prepended to the first user message
        img_content = []
        for img in sample[self._column_map["images"]]:
            img_content.append({"type": "image", "content": img})

        # Convert to messages
        messages = []
        for i, message in enumerate(sample[self._column_map["texts"]]):
            user_content = [{"type": "text", "content": message["user"]}]
            if i == 0:
                user_content = img_content + user_content
            messages.append(
                Message(
                    role="user",
                    content=user_content,
                    masked=True,
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

        return {"messages": messages}


# TODO: point to Flamingo model transform as an example
def the_cauldron_dataset(
    model_transform: Transform,
    *,
    subset: str,
    source: str = "HuggingFaceM4/the_cauldron",
    column_map: Optional[Dict[str, str]] = None,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> SFTDataset:
    """
    Support for family of image + text datasets similar to
    `The Cauldron <https://huggingface.co/datasets/HuggingFaceM4/the_cauldron>`_
    from Hugging Face Datasets.

    The Cauldron consists of numerous datasets. You must specify one of the datasets
    using the ``subset`` argument.

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
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.

    Returns:
        SFTDataset: dataset configured with source data and transform

    Raises:
        ValueError: If ``packed`` is True, they are not supported for multimodal datasets yet.

    Example:
        >>> cauldron_ds = the_cauldron_dataset(model_transform=model_transform, subset="ai2d")
        >>> for batch in Dataloader(cauldron_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    message_transform = TheCauldronToMessages(
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )

    ds = SFTDataset(
        model_transform=model_transform,
        source=source,
        message_transform=message_transform,
        name=subset,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )
    if packed:
        raise ValueError("Multimodal datasets don't support packing yet.")
    return ds
