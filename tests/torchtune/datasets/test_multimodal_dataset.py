# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Mapping
from unittest import mock

import torch
from datasets import Dataset
from tests.test_utils import DummyTokenizer
from torchtune.data import Message
from torchtune.datasets import MultimodalDataset
from torchvision.transforms.v2 import functional as F

SAMPLE = [
    {
        # This will be remapped from 0-1 to 0-255, so the image will be all 255s
        "images": F.to_pil_image(torch.ones(3, 10, 10)),
        "messages": [
            {
                "role": "user",
                "content": "What is in the center of the picture?",
                "masked": True,
            },
            {
                "role": "assistant",
                "content": "The center of the picture is a person.",
                "masked": False,
            },
            {
                "role": "user",
                "content": "What is the person doing?",
                "masked": True,
            },
            {
                "role": "assistant",
                "content": "The person is sitting on a bench.",
                "masked": False,
            },
        ],
    }
]


class DummyModelTransform:
    def __init__(self) -> None:
        self.tokenizer = DummyTokenizer()

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        tokens, mask = self.tokenizer.tokenize_messages(
            sample["messages"], max_seq_len=1000
        )
        sample.update(
            {
                "tokens": tokens,
                "mask": mask,
                "images": F.pil_to_tensor(sample["images"]).sum(),
            }
        )
        return sample


class DummyMessageTransform:
    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        sample["messages"] = [
            Message.from_dict(message) for message in sample["messages"]
        ]
        sample["messages"][0].content = [{"type": "image"}] + sample["messages"][
            0
        ].content
        return sample


class TestMultimodalDataset:
    @mock.patch("torchtune.datasets.multimodal._multimodal.load_dataset")
    def test_get_item(self, mock_load_dataset):
        mock_load_dataset.return_value = Dataset.from_list(SAMPLE)
        expected_tokenized_prompts = [
            [
                0,
                -2,
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
        expected_labels = [
            [
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                3,
                6,
                2,
                3,
                7,
                2,
                1,
                7,
                -1,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
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
            message_transform=DummyMessageTransform(),
            model_transform=DummyModelTransform(),
        )
        assert len(ds) == 1
        mock_load_dataset.assert_called_once()

        prompt, label = ds[0]["tokens"], ds[0]["labels"]
        assert prompt == expected_tokenized_prompts[0]
        assert label == expected_labels[0]

        assert ds[0]["images"].item() == 76500  # 255 * 3 * 10 * 10
