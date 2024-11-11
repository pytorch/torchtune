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

    def test_tokenization(self, tokenizer):
        texts = [
            "a cow jumping over the moon",
            "a helpful AI assistant",
        ]
        correct_tokens = [
            _pad(
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
                ]
            ),
            _pad(
                [2416, 320, 516, 75, 79, 69, 84, 331, 64, 328, 813, 667, 540, 339, 2417]
            ),
        ]
        tokens_tensor = tokenizer(texts)
        assert tokens_tensor.tolist() == correct_tokens

    def test_decoding(self, tokenizer):
        text = "this is torchtune"
        decoded_text = "<|startoftext|>this is torchtune <|endoftext|>"
        assert decoded_text == tokenizer.decode(tokenizer.encode(text))


def _pad(tokens, max_seq_len=77, pad_token=2417):
    while len(tokens) < max_seq_len:
        tokens.append(pad_token)
    return tokens
