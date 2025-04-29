# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.common import ASSETS
from tests.test_utils import DummyTokenizer
from torchtune.datasets.multimodal import vqa_dataset


class TestMultimodalInstructDataset:
    @pytest.fixture
    def tokenizer(self):
        return DummyTokenizer()

    def test_get_item(self, tokenizer):
        system_prompt = "follow this prompt"

        dataset = vqa_dataset(
            model_transform=tokenizer,
            source="json",
            data_files=str(ASSETS / "vqa_tiny.json"),
            split="train",
            new_system_prompt=system_prompt,
        )

        expected_tokens = [
            [0, 6, 4, 6, -2, 4, 2, 9, 2, 6, 7, 5, -1],
        ]

        expected_labels = [
            [-100, -100, -100, -100, -100, -100, -100, -100, -100, 7, 5, -1, -100]
        ]

        assert len(dataset) == 1

        for i in range(len(dataset)):
            prompt, label, image = (
                dataset[i]["tokens"],
                dataset[i]["labels"],
                dataset[i]["images"],
            )
            assert prompt == expected_tokens[i]
            assert label == expected_labels[i]
            assert isinstance(image[0], torch.Tensor)

    def test_dataset_fails_with_packed(self, tokenizer):
        with pytest.raises(
            ValueError, match="Multimodal datasets don't support packing yet."
        ):
            vqa_dataset(
                model_transform=tokenizer,
                source="json",
                packed=True,
            )
