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

from torchtune.datasets._hh_rlhf_helpful import hh_rlhf_helpful_dataset


class TestHHRLHFHelpfulDataset:
    @patch("torchtune.datasets._preference.load_dataset")
    @pytest.mark.parametrize("train_on_input", [True, False])
    def test_dataset_get_item(self, mock_load_dataset, train_on_input):
        # Truncated sample data from HH RLHF Helpful dataset
        mock_load_dataset.return_value = Dataset.from_list(
            [
                {
                    "chosen": [
                        {
                            "content": "helping my granny with her mobile phone issue",
                            "role": "user",
                        },
                        {
                            "content": "I see you are chatting with your grandmother "
                            "about an issue with her mobile phone. How can I help?",
                            "role": "assistant",
                        },
                        {"content": "her phone is not turning on", "role": "user"},
                        {
                            "content": "Is it on but it doesn’t power up or charge? "
                            "Or it’s off and does not turn on?",
                            "role": "assistant",
                        },
                    ],
                    "rejected": [
                        {
                            "content": "helping my granny with her mobile phone issue",
                            "role": "user",
                        },
                        {
                            "content": "I see you are chatting with your grandmother "
                            "about an issue with her mobile phone. How can I help?",
                            "role": "assistant",
                        },
                        {"content": "her phone is not turning on", "role": "user"},
                        {
                            "content": "Okay, are you concerned that her phone is broken, "
                            "or simply that it is not turning on?",
                            "role": "assistant",
                        },
                    ],
                }
            ]
        )
        ds = hh_rlhf_helpful_dataset(
            tokenizer=DummyTokenizer(),
            train_on_input=train_on_input,
        )
        # Generate the input and labels
        sample = ds[0]

        expected_chosen_counts = {
            3: 14,
            2: 11,
            4: 7,
            5: 7,
            7: 4,
            6: 4,
            0: 2,
            1: 2,
            -1: 2,
            8: 1,
            11: 1,
        }
        assert Counter(sample["chosen_input_ids"]) == expected_chosen_counts
        if train_on_input:
            assert Counter(sample["chosen_labels"]) == expected_chosen_counts
        else:
            # Check that the input is masked
            assert sample["chosen_labels"].count(CROSS_ENTROPY_IGNORE_IDX) == 16

        expected_rejected_counts = {
            3: 14,
            2: 8,
            5: 8,
            4: 6,
            6: 5,
            7: 4,
            0: 2,
            1: 2,
            -1: 2,
            8: 1,
            11: 1,
            9: 1,
        }
        assert Counter(sample["rejected_input_ids"]) == expected_rejected_counts
        if train_on_input:
            assert Counter(sample["rejected_labels"]) == expected_rejected_counts
        else:
            # Check that the input is masked
            assert sample["rejected_labels"].count(CROSS_ENTROPY_IGNORE_IDX) == 16
