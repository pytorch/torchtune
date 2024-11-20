# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest

from tests.common import ASSETS
from torchtune.models.clip._model_builders import clip_tokenizer


class TestCLIPTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return clip_tokenizer(ASSETS / "tiny_bpe_merges.txt")

    def test_encoding(self, tokenizer):
        texts = [
            "a cow jumping over the moon",
            "a helpful AI assistant",
        ]
        correct_tokens = [
            [
                2416,
                320,
                66,
                78,
                342,
                73,
                669,
                79,
                515,
                326,
                1190,
                337,
                673,
                324,
                76,
                819,
                333,
                2417,
            ],
            [2416, 320, 516, 75, 79, 69, 84, 331, 64, 328, 813, 667, 540, 339, 2417],
        ]
        for text, correct in zip(texts, correct_tokens):
            tokens = tokenizer.encode(text)
            assert tokens == correct

    def test_decoding(self, tokenizer):
        text = "this is torchtune"
        decoded_text = "<|startoftext|>this is torchtune <|endoftext|>"
        assert decoded_text == tokenizer.decode(tokenizer.encode(text))

    def test_call(self, tokenizer):
        sample = {"text": "hello world"}
        sample = tokenizer(sample)
        assert "text" not in sample
        assert "tokens" in sample
