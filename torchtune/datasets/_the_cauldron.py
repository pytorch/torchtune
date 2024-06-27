# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Mapping, Optional

from torchtune.data import Message
from torchtune.datasets._multimodal import MultimodalDataset
from torchtune.modules.transforms import Transform


def get_cauldron_messages(
    sample: Mapping[str, Any], train_on_input: bool = False
) -> List[Message]:
    """
    Construct messages from a sample formatted similarly to
    `The Cauldron dataset<https://huggingface.co/datasets/HuggingFaceM4/the_cauldron>_`.

    Images are including as a ``Message`` prepended to the user input.

    Args:
        sample (Mapping[str, Any]): A sample from the dataset.
        train_on_input (bool): Whether to train on the input or not.

    Returns:
        List[Message]: A list of messages representing the single sample
    """
    messages = []
    for message in sample["texts"]:
        messages.append(
            Message(
                role="user", content=message["user"], masked=not self.train_on_input
            )
        )
        messages.append(Message(role="assistant", content=message["assistant"]))
    # First message is a user message that should have the image attachment
    messages[0].media = ["image"]
    return messages


def the_cauldron_dataset(
    model_transform: Transform,
    *,
    source: str = "HuggingFaceM4/the_cauldron",
    chat_format: Optional[str] = None,
    train_on_input: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> MultimodalDataset:
    """
    Support for family of image + text datasets similar to
    `The Cauldron<https://huggingface.co/datasets/HuggingFaceM4/the_cauldron>_`
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

    return MultimodalDataset(
        model_transform=model_transform,
        source=source,
        convert_to_messages=get_cauldron_messages,
        chat_format=_get_component_from_path(chat_format)
        if chat_format is not None
        else None,
        train_on_input=train_on_input,
        **load_dataset_kwargs,
    )
