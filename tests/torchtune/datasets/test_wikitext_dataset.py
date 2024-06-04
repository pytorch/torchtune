# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from unittest.mock import patch

import pytest

from tests.test_utils import get_assets_path

from torchtune.datasets import wikitext_dataset
from torchtune.modules.tokenizers import SentencePieceTokenizer


class TestWikiTextDataset:
    @pytest.fixture
    def tokenizer(self):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        return SentencePieceTokenizer(str(get_assets_path() / "m.model"))

    @patch("torchtune.datasets._text_completion.load_dataset")
    @pytest.mark.parametrize("max_seq_len", [128, 512, 1024, 4096])
    def test_dataset_get_item(self, load_dataset, tokenizer, max_seq_len):
        # Sample data from wikitext dataset
        load_dataset.return_value = [
            {
                "text": "Bart , like the rest of his family , has yellow skin . "
                "Bart usually wears a red T @-@ shirt , blue shorts and blue trainers . "
                "When the Simpson family goes to church in the episodes , or to school "
                "events or shows , Bart wears a blue suit with a white shirt , a purple "
                "tie , blue shorts and a blue jacket .",
            }
        ]
        ds = wikitext_dataset(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        input, label = ds[0]["tokens"], ds[0]["labels"]
        assert len(input) <= max_seq_len
        assert len(label) <= max_seq_len
        assert len(input) == len(label)
        assert input[0] == tokenizer.bos_id
