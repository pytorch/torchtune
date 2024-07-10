# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest
from torchtune.data._types import Message
from torchtune.modules.tokenizers import TikTokenTokenizer

ASSETS = Path(__file__).parent.parent.parent.parent / "assets"


class TestTikTokenTokenizer:
    @pytest.fixture
    def tokenizer(self):
        # Pretrained tiktoken model generated via the script in
        # https://gist.github.com/ebsmothers/54b133dd87db6679b14318545aaa2de4
        return TikTokenTokenizer(str(ASSETS / "tiktoken_small.model"))

    @pytest.fixture
    def texts(self):
        return [
            "I can see the sun. But even if I cannot see the sun, I know that it exists.",
            "And to know that the sun is there - that is living.",
        ]

    @pytest.fixture
    def messages(self, texts):
        return [
            Message(role="user", content=texts[0], masked=True),
            Message(role="assistant", content=texts[1], masked=False),
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

    @pytest.fixture
    def tokenized_messages(self, token_ids):
        return (
            [2000, 2006, 477, 273, 2007, 10, 10]
            + token_ids
            + [
                2009,
                2006,
                520,
                511,
                446,
                2007,
                10,
                10,
                65,
                269,
                277,
                686,
                334,
                262,
                376,
                110,
                351,
                443,
                32,
                45,
                334,
                351,
                1955,
                46,
                2009,
                2001,
            ],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        )

    def test_encode(self, tokenizer, texts, token_ids):
        assert tokenizer.encode(texts[0], add_bos=True, add_eos=True) == [
            tokenizer.bos_id
        ] + token_ids + [tokenizer.eos_id]
        assert tokenizer.encode(texts[0], add_bos=False, add_eos=False) == token_ids

    def test_decode(self, tokenizer, texts, token_ids):
        assert tokenizer.decode(token_ids) == texts[0]

    def test_encode_and_decode(self, tokenizer, texts):
        token_ids = tokenizer.encode(texts[0], add_bos=True, add_eos=True)
        decoded_text = tokenizer.decode(token_ids)
        assert texts[0] == decoded_text

    def test_token_ids(self, tokenizer):
        assert tokenizer.bos_id == 2000
        assert tokenizer.eos_id == 2001
        assert tokenizer.pad_id == -1
        assert tokenizer.step_id == 2005
        assert tokenizer.start_header_id == 2006
        assert tokenizer.end_header_id == 2007
        assert tokenizer.eom_id == 2008
        assert tokenizer.eot_id == 2009
        assert tokenizer.python_tag == 2255

    def test_tokenizer_vocab_size(self, tokenizer):
        assert tokenizer.base_vocab_size == 2000
        assert tokenizer.vocab_size == 2256

    def test_tokenize_messages(self, tokenizer, messages, tokenized_messages):
        assert tokenizer.tokenize_messages(messages) == tokenized_messages
