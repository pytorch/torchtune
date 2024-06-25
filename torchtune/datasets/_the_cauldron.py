# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

from torchtune.data import Message

from torchtune.datasets._finetune import FinetuneDataset
from torchtune.modules.transforms import Transform


class CauldronMessages:
    def __init__(
        self,
        train_on_input: bool = False,
    ):
        self.train_on_input = train_on_input

    def __call__(
        self,
        sample: Dict[str, Any],
    ) -> List[Message]:

        messages = []
        messages.append(
            Message(role="user", content="", image=True, masked=not self.train_on_input)
        )
        for message in sample["texts"]:
            messages.append(
                Message(
                    role="user", content=message["user"], masked=not self.train_on_input
                )
            )
            messages.append(Message(role="assistant", content=message["assistant"]))
        return messages


def the_cauldron_dataset(
    transform: Transform,
    *,
    source: str = "HuggingFaceM4/the_cauldron",
    train_on_input: bool = False,
    **load_dataset_kwargs,
) -> FinetuneDataset:

    message_transform = CauldronMessages(train_on_input=train_on_input)

    return FinetuneDataset(
        source=source,
        message_transform=message_transform,
        model_transform=transform,
        split="train",
        **load_dataset_kwargs,
    )
