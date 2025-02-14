# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from torchtune.data._messages import ShareGPTToMessages
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms import Transform


def multimodal_chat_dataset(
    model_transform: Transform,
    *,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    image_tag: Optional[str] = None,
    image_dir: Optional[str] = None,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> SFTDataset:
    """
    Configure a text+image dataset with conversations between user and model assistant.

    This builder function can be used to configure a custom multimodal dataset directly from the yaml config
    as an alternative to :class:`~torchtune.datasets.SFTDataset`, as it is made to be config friendly.

    The dataset is expected to follow the ShareGPT format:

    .. code-block:: text

        |  conversations                     | image        |
        |------------------------------------|--------------|
        | [{"from": "human", "value": "Q1"}, | images/1.jpg |
        |  {"from": "gpt", "value": "A1"}]   |              |

    This will be converted to:

    .. code-block:: python

        messages = [
            Message(
                role="user",
                content=[
                    {"type": "image", "content": [<PIL.Image.Image>]},
                    {"type": "text", "content": "Q1"},
                ],
            ),
            Message(role="assistant", content="A1"),
        ]

    This list of messages is then tokenized for model training. Currently, only a single image per conversation sample
    is supported, and it is always added to the first user message.

    If your dataset is not in the ShareGPT format, we recommend creating a custom message transform and
    using it in a custom dataset builder function similar to :class:`~torchtune.datasets.multimodal_chat_dataset`.

    If your column names are different, use the ``column_map`` parameter to point
    towards the columns with the conversations and images.

    Args:
        model_transform (Transform): callable that applies model-specific pre-processing to the sample.
            This includes tokenization and any modality-specific transforms. It is expected to return at
            minimum ``"tokens"`` and ``"mask"`` keys.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text"), pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details.
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns ("conversations", "image")
            to the new column names in the dataset. Keys should be "conversations", "image" and values should
            be the new column names. If None, keep the default "conversations", "image".
            Default is None.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        image_tag (Optional[str]): placeholder tags in the text content of each message to be replaced by dictionaries
            indicating to the tokenizer where to place image tokens. If images are present and this is None,
            then will prepend image tokens to the first user message in the sample by default. If text-only, leave
            this as None. Default is None.
        image_dir (Optional[str]): path to the directory containing the images that is prepended to all image
            paths in the dataset. If None, assume images are available in current working directory or are located
            on a remote url. For text-only,leave as None. Default is None.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``,
            such as ``data_files`` or ``split``.

    Examples:

    ::

        my_dataset.json
        [
            {
                "dialogue": [
                    {
                        "from": "human",
                        "value": "<image>What time is it on the clock?",
                    },
                    {
                        "from": "gpt",
                        "value": "It is 10:00 AM.",
                    },
                ],
                "image_path": "images/clock.jpg",
            },
            ...,
        ]

    ::

        >>> from torchtune.models.llama3_2_vision import llama3_2_vision_transform
        >>> from torchtune.datasets.multimodal import multimodal_chat_dataset
        >>> model_transform = llama3_2_vision_transform(
        >>>     path="/tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model",
        >>>     prompt_template="torchtune.data.QuestionAnswerTemplate",
        >>>     max_seq_len=8192,
        >>>     image_size=560,
        >>> )
        >>> ds = multimodal_chat_dataset(
        >>>     model_transform=model_transform,
        >>>     source="json",
        >>>     data_files="data/my_data.json",
        >>>     column_map={
        >>>         "dialogue": "conversations",
        >>>         "image_path": "image",
        >>>     },
        >>>     image_dir="/home/user/dataset/",  # /home/user/dataset/images/clock.jpg
        >>>     image_tag="<image>",
        >>>     split="train",
        >>> )
        >>> tokenized_dict = ds[0]
        >>> print(model_transform.decode(tokenized_dict["tokens"], skip_special_tokens=False))
        >>> # '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nQuestion:<|image|>What time is it on the clock?Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nIt is 10:00AM.<|eot_id|>' # noqa
        >>> print(tokenized_dict["encoder_input"]["images"][0].shape)  # (num_tiles, num_channels, tile_height, tile_width)
        >>> # torch.Size([4, 3, 224, 224])

    This can also be accomplished via the yaml config:

    .. code-block:: yaml

        tokenizer:
          _component_: torchtune.models.llama3_2_vision_transform
          path: /tmp/Meta-Llama-3-8B-Instruct/original/tokenizer.model
          prompt_template: torchtune.data.QuestionAnswerTemplate
          max_seq_len: 8192

        dataset:
          _component_: torchtune.datasets.multimodal.multimodal_chat_dataset
          source: json
          data_files: data/my_data.json
          split: train
          column_map:
            dialogue: conversations
            image_path: image
          image_dir: /home/user/dataset/
          image_tag: "<image>"
          split: train

    Returns:
        SFTDataset: the configured :class:`~torchtune.datasets.SFTDataset`

    Raises:
        ValueError: If ``packed`` is True, they are not supported for multimodal datasets yet.

    """
    if packed:
        raise ValueError("Multimodal datasets don't support packing yet.")

    message_transform = ShareGPTToMessages(
        train_on_input=False,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
        image_dir=Path(image_dir),
        image_tag=image_tag,
    )

    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=model_transform,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )

    return ds
