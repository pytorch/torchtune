# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from unittest.mock import patch

import PIL

import pytest
from datasets import Dataset

from tests.test_utils import DummyTokenizer
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX

from torchtune.datasets.multimodal import llava_instruct_dataset


class TestLLaVAInstructDataset:
    @pytest.fixture
    def tokenizer(self):
        return DummyTokenizer()

    @pytest.fixture
    def test_image_pil(self):
        return PIL.Image.new(mode="RGB", size=(4, 4))

    @patch("torchtune.datasets._sft.load_dataset")
    @patch("torchtune.data._messages.load_image")
    def test_get_item(self, load_image, load_dataset, tokenizer, test_image_pil):
        """
        WARNING: careful with these mocks, they are applied in bottom up order
        """
        # mock the call to load_image
        load_image.return_value = test_image_pil

        # mock the call to HF datasets
        load_dataset.return_value = Dataset.from_list(
            [
                {
                    "image": "test_image.jpg",
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nWhat can you infer about the man's outdoor activity?",
                        },
                        {
                            "from": "gpt",
                            "value": "From the image, we can infer that the man is engaging in a "
                            "recreational activity involving a frisbee in a park or grass field. "
                            "The frisbee is in the air, and the man appears to be either catching "
                            "or throwing it. This suggests that he might be playing a casual game "
                            "of catch with a friend or practicing his frisbee skills, enjoying the "
                            "outdoors and getting some physical activity at the same time.",
                        },
                    ],
                }
            ]
        )

        ds = llava_instruct_dataset(
            model_transform=tokenizer,
        )

        input, labels, images = ds[0]["tokens"], ds[0]["labels"], ds[0]["images"]

        expected_count = {
            3: 17,
            2: 15,
            4: 11,
            8: 9,
            5: 8,
            7: 8,
            6: 5,
            1: 5,
            9: 2,
            0: 1,
            -2: 1,
            12: 1,
            10: 1,
            -1: 1,
        }

        assert Counter(input) == expected_count
        assert labels.count(CROSS_ENTROPY_IGNORE_IDX) == 11
        assert images == [test_image_pil]
