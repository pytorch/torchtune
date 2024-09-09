# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import PIL

import pytest
from tests.test_utils import DummyTokenizer
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX

from torchtune.datasets import the_cauldron_dataset


class TestTheCauldronDataset:
    @pytest.fixture
    def tokenizer(self):
        return DummyTokenizer()

    @pytest.fixture
    def test_image_pil(self):
        return PIL.Image.new(mode="RGB", size=(4, 4))

    @patch("torchtune.datasets._sft.load_dataset")
    def test_label_no_masking(self, load_dataset, tokenizer, test_image_pil):
        """
        Test whether the input and the labels are correctly created when the input is not masked.
        """
        # mock the call to HF datasets
        load_dataset.return_value = [
            {
                "images": [test_image_pil],
                "texts": [
                    {
                        "user": "Question: What do respiration and combustion give out"
                        "\nChoices:\nA. Oxygen\nB. Carbon dioxide\nC. Nitrogen\nD. Heat"
                        "\nAnswer with the letter.",
                        "assistant": "Answer: B",
                        "source": "AI2D",
                    }
                ],
            }
        ]

        ds = the_cauldron_dataset(
            model_transform=tokenizer, subset="dummy", train_on_input=True
        )
        input, labels = (
            ds[0]["tokens"],
            ds[0]["labels"],
        )

        assert input == [
            0,
            -2,
            9,
            4,
            2,
            11,
            3,
            10,
            4,
            3,
            8,
            2,
            6,
            2,
            6,
            7,
            2,
            8,
            2,
            4,
            6,
            4,
            3,
            7,
            7,
            1,
            -1,
        ]
        assert labels == input

    @patch("torchtune.datasets._sft.load_dataset")
    def test_label_masking(self, load_dataset, tokenizer, test_image_pil):
        """
        Test whether the input and the labels are correctly created when the input is masked.
        """
        # mock the call to HF datasets
        load_dataset.return_value = [
            {
                "images": [test_image_pil],
                "texts": [
                    {
                        "user": "Question: What do respiration and combustion give out"
                        "\nChoices:\nA. Oxygen\nB. Carbon dioxide\nC. Nitrogen\nD. Heat"
                        "\nAnswer with the letter.",
                        "assistant": "Answer: B",
                        "source": "AI2D",
                    }
                ],
            }
        ]

        ds = the_cauldron_dataset(
            model_transform=tokenizer, subset="dummy", train_on_input=False
        )
        input, labels = (
            ds[0]["tokens"],
            ds[0]["labels"],
        )

        assert input == [
            0,
            -2,
            9,
            4,
            2,
            11,
            3,
            10,
            4,
            3,
            8,
            2,
            6,
            2,
            6,
            7,
            2,
            8,
            2,
            4,
            6,
            4,
            3,
            7,
            7,
            1,
            -1,
        ]
        assert labels.count(CROSS_ENTROPY_IGNORE_IDX) == 24
