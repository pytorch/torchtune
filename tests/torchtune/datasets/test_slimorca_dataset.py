# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from unittest.mock import patch

import pytest
from datasets import Dataset

from tests.test_utils import DummyTokenizer

from torchtune.datasets import slimorca_dataset


class TestSlimOrcaDataset:
    @pytest.fixture
    def tokenizer(self):
        return DummyTokenizer()

    @patch("torchtune.datasets._chat.load_dataset")
    def test_value_error(self, load_dataset, tokenizer):
        load_dataset.return_value = []
        with pytest.raises(ValueError):
            slimorca_dataset(tokenizer=tokenizer, max_seq_len=3)

    @patch("torchtune.datasets._chat.load_dataset")
    @pytest.mark.parametrize("max_seq_len", [128, 512, 1024, 4096])
    def test_dataset_get_item(self, load_dataset, tokenizer, max_seq_len):
        # Sample data from slimorca dataset
        load_dataset.return_value = Dataset.from_list(
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
            max_seq_len=max_seq_len,
            train_on_input=(max_seq_len == 128),
        )
        input, label = ds[0]["tokens"], ds[0]["labels"]
        assert len(input) <= max_seq_len
        assert len(label) <= max_seq_len
        assert len(input) == len(label)
        assert input[0] == tokenizer.bos_id
        assert input[-1] == tokenizer.eos_id
        assert label[-1] == tokenizer.eos_id
