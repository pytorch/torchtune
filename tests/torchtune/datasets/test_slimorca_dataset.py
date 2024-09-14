# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from collections import Counter
from unittest.mock import patch

import pytest
from datasets import Dataset

from tests.test_utils import DummyTokenizer
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX

from torchtune.datasets import slimorca_dataset


class TestSlimOrcaDataset:
    @pytest.fixture
    def tokenizer(self):
        return DummyTokenizer()

    @patch("torchtune.datasets._sft.load_dataset")
    @pytest.mark.parametrize("train_on_input", [True, False])
    def test_dataset_get_item(self, mock_load_dataset, train_on_input, tokenizer):
        # Sample data from slimorca dataset
        mock_load_dataset.return_value = Dataset.from_list(
            [
                {
                    "conversations": [
                        {
                            "from": "system",
                            "value": "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",  # noqa: B950
                        },
                        {
                            "from": "human",
                            "value": "Please briefly summarize this news article:\n\nAOL.com Video - Father Lets 8-Year-Old Drive On Icy Road\n\nDescription:Would you let your 8-year-old drive your car? How about on an icy road? Well one father in Russia did just that, and recorded the entire thing. To her credit, the child seemed to be doing a great job. (0:44)\n\nTags: 8-year-old driver , caught on camera , child driver , pix11\n\nSummary:",  # noqa: B950
                        },
                        {
                            "from": "gpt",
                            "value": "A father in Russia allowed his 8-year-old child to drive his car on an icy road and recorded the event. The child appeared to be handling the situation well, showcasing their driving skills despite the challenging conditions.",  # noqa: B950
                        },
                    ]
                }
            ]
        )
        ds = slimorca_dataset(
            tokenizer=tokenizer,
            train_on_input=train_on_input,
        )
        # Generate the input and labels
        input, labels = ds[0]["tokens"], ds[0]["labels"]

        expected_counts = {
            3: 28,
            2: 20,
            4: 20,
            5: 20,
            6: 17,
            10: 8,
            1: 7,
            8: 7,
            7: 7,
            9: 2,
            11: 2,
            0: 1,
            12: 1,
            17: 1,
            -1: 1,
        }
        assert Counter(input) == expected_counts
        if train_on_input:
            assert Counter(labels) == expected_counts
        else:
            # Check that the input is masked
            assert labels.count(CROSS_ENTROPY_IGNORE_IDX) == 104
