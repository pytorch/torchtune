# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest

from torchtune.models.clip._model_builders import clip_tokenizer


class TestCLIPTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return clip_tokenizer()

    def test_tokenization(self, tokenizer):
        texts = [
            "a cow jumping over the moon",
            "a helpful AI assistant",
        ]
        correct_tokens = [
            [49406, 320, 9706, 11476, 962, 518, 3293, 49407],
            [49406, 320, 12695, 2215, 6799, 49407],
        ]
        for token_seq in correct_tokens:
            _pad_token_sequence(token_seq)
        tokens_tensor = tokenizer(texts)
        assert tokens_tensor.tolist() == correct_tokens

    def test_text_cleaning(self, tokenizer):
        text = "(à¸‡'âŒ£')à¸‡"
        correct_tokens = [49406, 263, 33382, 6, 17848, 96, 19175, 33382, 49407]
        tokens = tokenizer.encode(text)
        assert tokens == correct_tokens

    def test_decoding(self, tokenizer):
        text = "this is torchtune"
        decoded_text = "<|startoftext|>this is torchtune <|endoftext|>"
        assert decoded_text == tokenizer.decode(tokenizer.encode(text))


def _pad_token_sequence(tokens, max_seq_len=77, pad_token=49407):
    while len(tokens) < max_seq_len:
        tokens.append(pad_token)
