# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest
from torchtune.modules.tokenizer import Tokenizer

ASSETS = Path(__file__).parent.parent.parent / "assets"


class TestTokenizer:
    @pytest.fixture
    def tokenizer(self):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        return Tokenizer.from_file(str(ASSETS / "m.model"))

    @pytest.fixture
    def tokenizer_with_max_len(self, request):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        return Tokenizer.from_file(str(ASSETS / "m.model"), request.param)

    @pytest.mark.parametrize(
        "tokenizer_with_max_len", [4, 100], indirect=["tokenizer_with_max_len"]
    )
    def test_truncation(self, tokenizer_with_max_len):
        token_list = [
            tokenizer_with_max_len.bos_id,
            12,
            1803,
            1024,
            103,
            tokenizer_with_max_len.eos_id,
        ]
        if tokenizer_with_max_len.max_len < 6:
            token_list = token_list[: tokenizer_with_max_len.max_len]
        assert (
            tokenizer_with_max_len.encode("Hello world!", truncate=True) == token_list
        )

    def test_encode(self, tokenizer):
        assert tokenizer.encode("Hello world!") == [
            tokenizer.bos_id,
            12,
            1803,
            1024,
            103,
            tokenizer.eos_id,
        ]
        assert tokenizer.encode("Hello world!", add_eos=False) == [
            tokenizer.bos_id,
            12,
            1803,
            1024,
            103,
        ]
        assert tokenizer.encode("Hello world!", add_bos=False) == [
            12,
            1803,
            1024,
            103,
            tokenizer.eos_id,
        ]
        assert tokenizer.encode("Hello world!", add_eos=False, add_bos=False) == [
            12,
            1803,
            1024,
            103,
        ]

    def test_decode(self, tokenizer):
        assert tokenizer.decode([1, 12, 1803, 1024, 103, 2]) == "Hello world!"

    def test_token_ids(self, tokenizer):
        assert tokenizer.eos_id == 2
        assert tokenizer.pad_id == -1
        assert tokenizer.bos_id == 1

    def test_tokenizer_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 2000
