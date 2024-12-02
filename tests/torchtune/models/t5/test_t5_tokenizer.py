# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest

from tests.common import ASSETS
from torchtune.models.t5._model_builders import t5_tokenizer


class TestT5Tokenizer:
    @pytest.fixture
    def tokenizer(self):
        return t5_tokenizer(str(ASSETS / "sentencepiece.model"))

    def test_encoding(self, tokenizer):
        texts = [
            "a cow jumping over the moon",
            "a helpful AI assistant",
        ]
        correct_tokens = [
            [3, 9, 9321, 15539, 147, 8, 8114, 1],
            [3, 9, 2690, 7833, 6165, 1],
        ]
        for text, correct in zip(texts, correct_tokens):
            tokens = tokenizer.encode(text)
            print(tokens)
            assert tokens == correct

    def test_decoding(self, tokenizer):
        text = "this is torchtune"
        assert text == tokenizer.decode(tokenizer.encode(text))

    def test_call(self, tokenizer):
        sample = {"text": "hello world"}
        sample = tokenizer(sample)
        assert "text" not in sample
        assert "tokens" in sample
