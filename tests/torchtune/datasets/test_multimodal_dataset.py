# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping
from unittest import mock

import torch
from datasets import Dataset
from PIL.Image import Image
from tests.test_utils import DummyChatFormat, DummyTokenizer
from torchtune.data import Message
from torchtune.datasets import MultimodalDataset
from torchvision.transforms.v2 import functional as F

SAMPLE = [
    {
        "images": F.to_pil_image(torch.ones(3, 10, 10)),
        "messages": [
            {
                "role": "user",
                "content": "What is in the center of the picture?",
            },
            {
                "role": "assistant",
                "content": "The center of the picture is a person.",
            },
            {
                "role": "user",
                "content": "What is the person doing?",
            },
            {
                "role": "assistant",
                "content": "The person is sitting on a bench.",
            },
        ],
    }
]


class DummyModelTransform:
    def __init__(self) -> None:
        self.tokenizer = DummyTokenizer()

    def __call__(
        self, *, images: Image, messages: List[Message], **kwargs
    ) -> Mapping[str, Any]:
        messages = [Message.from_image(images)] + messages
        tokenized_dict = self.tokenizer.tokenize_messages(messages, max_seq_len=1000)
        tokenized_dict["images"] = F.pil_to_tensor(tokenized_dict["images"]).sum()
        kwargs.update(tokenized_dict)
        return kwargs


class TestMultimodalDataset:
    @mock.patch("torchtune.datasets._multimodal.load_dataset")
    def test_get_item(self, mock_load_dataset):
        mock_load_dataset.return_value = Dataset.from_list(SAMPLE)
        expected_tokenized_prompts = [
            [
                0,
                4,
                2,
                2,
                3,
                6,
                2,
                3,
                8,
                3,
                6,
                2,
                3,
                7,
                2,
                1,
                7,
                -1,
                0,
                4,
                2,
                3,
                6,
                6,
                3,
                6,
                2,
                7,
                2,
                1,
                6,
                -1,
            ],
        ]
        ds = MultimodalDataset(
            source="iam/agoofy/goober",
            convert_to_messages=lambda x, y: x["messages"],
            chat_format=DummyChatFormat,
            model_transform=DummyModelTransform(),
            train_on_input=False,
        )
        assert len(ds) == 1
        mock_load_dataset.assert_called_once()

        prompt, label = ds[0]["tokens"], ds[0]["labels"]
        assert prompt == expected_tokenized_prompts[0]
        assert label == expected_tokenized_prompts[0]

        assert ds[0]["image"].item() == 300
