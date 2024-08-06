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

from torchtune.datasets import stack_exchange_paired_dataset


class TestStackExchangePairedDataset:
    @patch("torchtune.datasets._preference.load_dataset")
    @pytest.mark.parametrize("train_on_input", [True, False])
    def test_dataset_get_item(self, mock_load_dataset, train_on_input):
        # Truncated sample data from stack exchange paired dataset
        mock_load_dataset.return_value = Dataset.from_list(
            [
                {
                    "question": "I have a question about if a animation ends that it "
                    "will like `gotoAndStop()` to another frame ``` if (bird.hitTestObject(pipe1))"
                    " { bird.gotoAndStop(3); //frame 3 = animation } ``` after it ends it will need"
                    " to go the Game Over frame (frame 3) and I use the `Flash Timeline` not `.as` "
                    "thanks!",
                    "response_j": "Java does not provide a convenient way to list the 'files' "
                    "in a 'directory', when that directory is backed by a JAR file on the classpath"
                    " (see [How do I list the files inside a JAR file?](https://stackoverflow.com/"
                    "questions/1429172/how-do-i-list-the-files-inside-a-jar-file) for some work-arounds)",
                    "response_k": "If you are still looking for an actual answer here is [mine]"
                    "(https://pastebin.com/R0jMh4ui) (it is kinda hacky but its work). To use it "
                    "you simply have to call one of the 2 options below",
                }
            ]
        )
        ds = stack_exchange_paired_dataset(
            tokenizer=DummyTokenizer(),
            train_on_input=train_on_input,
        )
        # Generate the input and labels
        sample = ds[0]

        expected_chosen_counts = {
            4: 20,
            2: 15,
            3: 15,
            1: 13,
            9: 6,
            5: 6,
            7: 6,
            6: 4,
            0: 1,
            8: 1,
            15: 1,
            27: 1,
            20: 1,
            10: 1,
            12: 1,
            93: 1,
            13: 1,
            -1: 1,
        }
        assert Counter(sample["chosen_input_ids"]) == expected_chosen_counts
        if train_on_input:
            assert Counter(sample["chosen_labels"]) == expected_chosen_counts
        else:
            # Check that the input is masked
            assert sample["chosen_labels"].count(CROSS_ENTROPY_IGNORE_IDX) == 54

        expected_rejected_counts = {
            2: 17,
            3: 17,
            4: 13,
            1: 9,
            5: 9,
            7: 6,
            6: 6,
            9: 4,
            0: 1,
            8: 1,
            15: 1,
            27: 1,
            20: 1,
            37: 1,
            -1: 1,
        }
        assert Counter(sample["rejected_input_ids"]) == expected_rejected_counts
        if train_on_input:
            assert Counter(sample["rejected_labels"]) == expected_rejected_counts
        else:
            # Check that the input is masked
            assert sample["rejected_labels"].count(CROSS_ENTROPY_IGNORE_IDX) == 54
