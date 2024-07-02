# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest
from torchtune.data._types import Message
from torchtune.models.llama3 import llama3_tokenizer, Llama3Tokenizer

ASSETS = Path(__file__).parent.parent.parent.parent / "assets"


class TestLlama3Tokenizer:
    @pytest.fixture
    def tokenizer(self):
        # Pretrained tiktoken model generated via the script in
        # https://gist.github.com/ebsmothers/54b133dd87db6679b14318545aaa2de4
        return llama3_tokenizer(
            path=str(ASSETS / "tiktoken_small.model"),
        )

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
            [128000, 128006, 477, 273, 128007, 10, 10]
            + token_ids
            + [
                128009,
                128006,
                520,
                511,
                446,
                128007,
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
                128009,
                128001,
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

    def test_token_ids(self, tokenizer):
        assert tokenizer.bos_id == 128000
        assert tokenizer.eos_id == 128001
        assert tokenizer.pad_id == 0
        assert tokenizer.start_header_id == 128006
        assert tokenizer.end_header_id == 128007
        assert tokenizer.eom_id == 128008
        assert tokenizer.eot_id == 128009
        assert tokenizer.python_tag == 128255

    def test_tokenizer_vocab_size(self, tokenizer):
        assert tokenizer.base_vocab_size == 2000
        assert tokenizer.vocab_size == 128256

    def test_tokenize_messages(self, tokenizer, messages, tokenized_messages):
        assert tokenizer.tokenize_messages(messages) == tokenized_messages

    def test_validate_special_tokens(self):
        with pytest.raises(
            ValueError, match="<|begin_of_text|> missing from special_tokens"
        ):
            tokenizer = Llama3Tokenizer(
                path=str(ASSETS / "tiktoken_small.model"),
                # Same as LLAMA3_SPECIAL_TOKENS but one missing
                special_tokens={
                    "<|end_of_text|>": 128001,
                    "<|start_header_id|>": 128006,
                    "<|end_header_id|>": 128007,
                    "<|eot_id|>": 128009,
                    "<|eom_id|>": 128008,
                    "<|python_tag|>": 128255,
                },
            )
