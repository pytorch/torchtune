# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# flake8: noqa: E128

import pytest

from tests.common import ASSETS
from torchtune.data import Message
from torchtune.models.phi4 import phi4_tokenizer


class TestPhi4Tokenizer:
    @pytest.fixture
    def tokenizer(self):
        # GPT2BaseTokenizer
        return phi4_tokenizer(
            vocab_path=(ASSETS / "vocab.json"),
            merges_path=(ASSETS / "merges.txt"),
        )

    @pytest.fixture
    def expected_tokens(self):
        # fmt: off
        tokens = [100264, 9125, 100266, 2675, 527, 264, 11190, 18328, 100265, 100264, 882, 100266, 14149, 28514, 374, 279, 1888, 6875, 0, 100265, 100264, 78191, 100266, 9642, 11, 433, 374, 0, 100265]
        # fmt: on
        return tokens

    def test_tokenize_messages(self, tokenizer, expected_tokens):
        messages = [
            Message(role="system", content="You are a helpful assistant", masked=True),
            Message(
                role="user",
                content="Pytorch is the best library!",
                masked=True,
            ),
            Message(
                role="assistant",
                content="Yes, it is!",
            ),
        ]
        tokens, mask = tokenizer.tokenize_messages(messages, add_eos=True)

        expected_mask = [True] * 20 + [False] * 9
        assert expected_tokens == tokens
        assert expected_mask == mask

    def test_tokenize_messages_no_system_prompt(self, tokenizer):
        messages = [
            Message(role="system", content="You are a helpful assistant", masked=True),
            Message(
                role="user",
                content="Pytorch is the best library!",
                masked=True,
            ),
            Message(
                role="assistant",
                content="Yes, it is!",
            ),
        ]
        tokens, mask = tokenizer.tokenize_messages(
            messages, ignore_system_prompt=True, add_eos=True
        )

        # fmt: off
        expected_tokens = [100264, 882, 100266, 14149, 28514, 374, 279, 1888, 6875, 0, 100265, 100264, 78191, 100266, 9642, 11, 433, 374, 0, 100265]
        # fmt: on

        expected_mask = [True] * 11 + [False] * 9
        assert expected_tokens == tokens
        assert expected_mask == mask

    def test_tokenize_message_drop_eos(self, tokenizer, expected_tokens):
        """
        Test that the tokenizer will not add an EOS token or EOT token if user requests it.
        This is the most common case for inference.
        """
        messages = [
            Message(role="system", content="You are a helpful assistant", masked=True),
            Message(
                role="user",
                content="Pytorch is the best library!",
                masked=True,
            ),
            Message(
                role="assistant",
                content="Yes, it is!",
            ),
        ]

        tokens, mask = tokenizer.tokenize_messages(messages, add_eos=False)
        expected_mask = [True] * 20 + [False] * 8
        # Drop eos token.
        assert expected_tokens[:-1] == tokens
        assert expected_mask == mask
