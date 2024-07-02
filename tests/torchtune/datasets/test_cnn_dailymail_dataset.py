# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from unittest.mock import patch

import pytest

from tests.test_utils import DummyTokenizer

from torchtune.datasets import cnn_dailymail_articles_dataset


class TestCNNDailyMailArticlesDataset:
    @pytest.fixture
    def tokenizer(self):
        return DummyTokenizer()

    @patch("torchtune.datasets._text_completion.load_dataset")
    @pytest.mark.parametrize("max_seq_len", [128, 512, 1024, 4096])
    def test_dataset_get_item(self, load_dataset, tokenizer, max_seq_len):
        # Sample data from CNN / DailyMail dataset
        load_dataset.return_value = [
            {
                "article": "(CNN) -- An American woman died aboard a cruise ship "
                "that docked at Rio de Janeiro on Tuesday, the same ship on which "
                "86 passengers previously fell ill, according to the state-run "
                "Brazilian news agency, Agencia Brasil. The American tourist died "
                "aboard the MS Veendam, owned by cruise operator Holland America. "
                "Federal Police told Agencia Brasil that forensic doctors were "
                "investigating her death. The ship's doctors told police that the "
                "woman was elderly and suffered from diabetes and hypertension, "
                "according the agency. The other passengers came down with diarrhea "
                "prior to her death during an earlier part of the trip, the ship's "
                "doctors said. The Veendam left New York 36 days ago for a South "
                "America tour.",
            }
        ]
        ds = cnn_dailymail_articles_dataset(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        input, label = ds[0]["tokens"], ds[0]["labels"]
        assert len(input) <= max_seq_len
        assert len(label) <= max_seq_len
        assert len(input) == len(label)
        assert input[0] == tokenizer.bos_id
