# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, Optional

from torchtune.config._utils import _get_component_from_path

from torchtune.data import ChatFormat, Message, split_text_by_image_tag
from torchtune.datasets.multimodal._multimodal import MultimodalDataset
from torchtune.modules.transforms import Transform


class LlavaInstructTransform(Transform):
    """
    Construct messages from a sample formatted similarly to the
    `LLaVA-Instruct-150K dataset<https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K>_`.

    The image tags <image> are replaced with placeholders in the ``Message`` content that
    the model tokenizer will replace with its image special token id.

    Attributes:
        train_on_input (bool): Whether to train on the input or not.

    Args:
        sample (Mapping[str, Any]): A sample from the dataset.

    Returns:
        Mapping[str, Any]: updated sample dict with the following keys:
            - messages (List[Message]): A list of messages representing the single sample
    """

    def __init__(
        self, train_on_input: bool = False, chat_format: Optional[ChatFormat] = None
    ):
        self.train_on_input = train_on_input
        self.chat_format = chat_format

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        conversations = sample["conversations"]

        messages = []
        for message in conversations:
            role = role_map[message["from"]]
            content = (
                split_text_by_image_tag(message["value"])
                if role == "user"
                else [{"type": "text", "content": message["value"]}]
            )
            masked = (role != "assistant") and (not self.train_on_input)
            messages.append(Message(role=role, content=content, masked=masked))
        return messages


def llava_instruct_dataset(
    model_transform: Transform,
    *,
    source: str = "liuhaotian/LLaVA-Instruct-150K",
    chat_format: Optional[str] = None,
    train_on_input: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> MultimodalDataset:
    """
    Support for family of image + text datasets similar to
    `LLaVA-Instruct-150K <https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K>_`
    from Hugging Face Datasets.

    Args:
        model_transform (Transform): model-specific transform that takes in a sample dict and applies custom
            transforms on the keys. The tokenizer used by the model should be encapsulated in the model transform
            and should operate on the "messages" field. The keys returned by the model should be aligned with the
            expected inputs into the model.
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path).
            Current default is path to The Cauldron, but can be configured for any similar dataset.
        chat_format (Optional[str]): name of template used to format the chat. See the description
            in :class:`~torchtune.datasets.ChatDataset` for more details. Default: None
        train_on_input (bool): Whether to train on the input or not. Default is False.
        **load_dataset_kwargs: Additional keyword arguments to pass to ``load_dataset``.

    Returns:
        MultimodalDataset: the configured :class:`~torchtune.datasets.MultimodalDataset`
    """

    message_transform = LlavaInstructTransform(
        train_on_input=train_on_input, chat_format=_get_component_from_path(chat_format)
    )

    return MultimodalDataset(
        model_transform=model_transform,
        source=source,
        message_transform=message_transform,
        split="train",
        **load_dataset_kwargs,
    )
