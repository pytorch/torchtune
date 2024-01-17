# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path

import pytest

from torchtune import datasets
from torchtune.modules.tokenizer import Tokenizer

ASSETS = Path(__file__).parent.parent.parent / "assets"


class TestSlimOrcaDataset:
    @pytest.fixture
    def tokenizer(self):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        return Tokenizer.from_file(str(ASSETS / "m.model"))

    def test_slim_orca_dataset(self, tokenizer):
        dataset = datasets.get_dataset("slimorca", tokenizer=tokenizer)
        assert dataset
