# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping
from unittest import mock

import torch
from datasets import Dataset
from tests.test_utils import DummyTokenizer
from torchtune.data import Message
from torchtune.datasets import FinetuneDataset

SAMPLE = [
    {
        "image": torch.ones(10, 10),
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


class DummyMessageTransform:
    def __call__(self, sample: Mapping[str, Any]) -> List[Message]:
        return [Message(role="user", image=True)] + [
            Message.from_dict(message) for message in sample["messages"]
        ]


class DummyModelTransform:
    def __init__(self) -> None:
        self.tokenizer = DummyTokenizer()

    def __call__(
        self, *, image: torch.Tensor, messages: List[Message], **kwargs
    ) -> Mapping[str, Any]:
        tokens, mask = self.tokenizer.tokenize_messages(messages, max_seq_len=1000)
        kwargs.update(
            {"tokens": tokens, "image": torch.tensor(image).sum(), "mask": mask}
        )
        return kwargs


class TestFinetuneDataset:
    @mock.patch("torchtune.datasets._finetune.load_dataset")
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
        ds = FinetuneDataset(
            source="iam/agoofy/goober",
            message_transform=DummyMessageTransform(),
            model_transform=DummyModelTransform(),
        )
        assert len(ds) == 1
        mock_load_dataset.assert_called_once()

        prompt, label = ds[0]["tokens"], ds[0]["labels"]
        assert prompt == expected_tokenized_prompts[0]
        assert label == expected_tokenized_prompts[0]

        assert ds[0]["image"].item() == 100
