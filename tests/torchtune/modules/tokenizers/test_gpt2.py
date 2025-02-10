# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from tests.common import ASSETS
from torchtune.modules.transforms.tokenizers import GPT2BaseTokenizer


class TestGPT2BaseTokenizer:
    @pytest.fixture
    def tokenizer(self):
        tokenizer = GPT2BaseTokenizer(
            ASSETS / "tiny_vocab.json",
            ASSETS / "tiny_bpe_merges.txt",
            "replace",
            1,
            1,
            1,
            1,
        )
        return tokenizer

    def test_encode(self, tokenizer):
        assert tokenizer.encode("Hello world!") == [
            tokenizer.bos_id,
            2,
            3,
            4,
            5,
            tokenizer.eos_id,
        ]
        assert tokenizer.encode("Hello world!", add_eos=False) == [
            tokenizer.bos_id,
            2,
            3,
            4,
            5,
        ]
        assert tokenizer.encode("Hello world!", add_bos=False) == [
            2,
            3,
            4,
            5,
            tokenizer.eos_id,
        ]
        assert tokenizer.encode("Hello world!", add_eos=False, add_bos=False) == [
            2,
            3,
            4,
            5,
        ]

    def test_decode(self, tokenizer):
        tokens = [2, 3, 4, 5]

        assert tokenizer.decode(tokens) == ["H", "ell", "o", "Ġworld"]
        assert tokenizer.decode(
            tokenizer.encode("Hello world!", add_eos=False, add_bos=False)
        ) == ["H", "ell", "o", "Ġworld"]
        assert tokenizer.decode(tokenizer.encode("Hello world!")) == [
            None,
            "H",
            "ell",
            "o",
            "Ġworld",
            None,
        ]

    def test_token_ids(self, tokenizer):
        assert tokenizer.eos_id == 1
        assert tokenizer.pad_id == 1
        assert tokenizer.bos_id == 1
        assert tokenizer.unk_id == 1

    def test_tokenizer_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 4
