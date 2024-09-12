# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from tests.common import ASSETS
from torchtune.modules.tokenizers import SentencePieceBaseTokenizer


class TestSentencePieceBaseTokenizer:
    @pytest.fixture
    def tokenizer(self):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        sp_tokenizer = SentencePieceBaseTokenizer(str(ASSETS / "m.model"))
        return sp_tokenizer

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

    def test_encode_without_leading_whitespace(self, tokenizer):
        s1 = "Hello"
        s2 = "I'm an outgoing and friendly person."
        # TODO: investigate why test tokenizer model does not encode whitespace
        tokenizer.encodes_whitespace = True
        s1_tokens = tokenizer.encode(s1, add_bos=False, add_eos=False)
        s2_tokens = tokenizer.encode(s2, add_bos=False, add_eos=False)
        # Set prefix="pre" since "\n" is not in the test tokenizer's vocab
        s2_tokens_no_whitespace = tokenizer.encode(
            s2, add_bos=False, add_eos=False, trim_leading_whitespace=True, prefix="pre"
        )
        s1s2_tokens = tokenizer.encode(s1 + s2, add_bos=False, add_eos=False)
        assert (s1_tokens + s2_tokens) != s1s2_tokens
        assert (s1_tokens + s2_tokens_no_whitespace) == s1s2_tokens
