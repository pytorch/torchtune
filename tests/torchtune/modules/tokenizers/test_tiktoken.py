# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from tests.common import ASSETS
from torchtune.models.llama3._tokenizer import CL100K_PATTERN
from torchtune.modules.tokenizers import TikTokenBaseTokenizer


class TestTikTokenBaseTokenizer:
    @pytest.fixture
    def tokenizer(self):
        # Pretrained tiktoken model generated via the script in
        # https://gist.github.com/ebsmothers/54b133dd87db6679b14318545aaa2de4
        return TikTokenBaseTokenizer(
            path=str(ASSETS / "tiktoken_small.model"),
            name="test_tiktoken",
            pattern=CL100K_PATTERN,
            bos_id=0,
            eos_id=-1,
            special_tokens={
                "<|test_token_0|>": 2000,
                "<|test_token_1|>": 2001,
            },
        )

    @pytest.fixture
    def texts(self):
        return [
            "I can see the sun. But even if I cannot see the sun, I know that it exists.",
            "And to know that the sun is there - that is living.",
        ]

    @pytest.fixture
    def token_ids(self):
        return [
            73,
            503,
            654,
            262,
            376,
            110,
            46,
            690,
            720,
            428,
            270,
            1119,
            654,
            262,
            376,
            110,
            44,
            270,
            686,
            334,
            312,
            522,
            511,
            115,
            46,
        ]

    def test_encode(self, tokenizer, texts, token_ids):
        assert tokenizer.encode(texts[0], add_bos=True, add_eos=True) == [
            0
        ] + token_ids + [-1]

    def test_decode(self, tokenizer, texts, token_ids):
        assert tokenizer.decode(token_ids) == texts[0]

    def test_encode_and_decode(self, tokenizer, texts):
        token_ids = tokenizer.encode(texts[0], add_bos=False, add_eos=False)
        decoded_text = tokenizer.decode(token_ids)
        assert texts[0] == decoded_text

    def test_tokenizer_vocab_size(self, tokenizer):
        assert tokenizer.base_vocab_size == 2000
        assert tokenizer.vocab_size == 2002

    def test_split_long_repetitions(self, tokenizer):
        normal_str = "Here is a normal string"
        ten_spaces = "".join(10 * [" "])
        space_str = ten_spaces.join(
            ["Here", "is", "a", "string", "with", "long", "spaces"]
        )
        no_space_str = "".join(10 * ["ab"])

        actual_split = tokenizer._split_long_repetitions(normal_str, 5)
        expected_split = ["Here is a norma", "l strin", "g"]
        for actual_substr, expected_substr in zip(actual_split, expected_split):
            assert actual_substr == expected_substr
        with pytest.raises(StopIteration):
            next(actual_split)

        actual_split = tokenizer._split_long_repetitions(space_str, 9)
        expected_split = [
            "Here" + ten_spaces[:-1],
            " is" + ten_spaces[:-1],
            " a" + ten_spaces[:-1],
            " string" + ten_spaces[:-1],
            " with" + ten_spaces[:-1],
            " long" + ten_spaces[:-1],
            " spaces",
        ]
        for actual_substr, expected_substr in zip(actual_split, expected_split):
            assert actual_substr == expected_substr
        with pytest.raises(StopIteration):
            next(actual_split)

        actual_split = tokenizer._split_long_repetitions(no_space_str, 4)
        expected_split = ["abab"] * 5
        for actual_substr, expected_substr in zip(actual_split, expected_split):
            assert actual_substr == expected_substr
        with pytest.raises(StopIteration):
            next(actual_split)
