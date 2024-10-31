# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.common import ASSETS

from torchtune.data import Message
from torchtune.models.qwen2_5 import qwen2_5_tokenizer


class TestQwen2_5Tokenizer:  # noqa: N801
    def tokenizer(self):
        return qwen2_5_tokenizer(
            path=str(ASSETS / "tiny_bpe_vocab.json"),
            merges_file=str(ASSETS / "tiny_bpe_merges.txt"),
        )

    def test_tokenize_messages(self):
        tokenizer = self.tokenizer()
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Give me a short introduction to LLMs."),
            Message(role="assistant", content=""),
        ]
        expected_tokens = [
            151644,
            82,
            88,
            479,
            94,
            56,
            119,
            230,
            98,
            374,
            494,
            1318,
            249,
            13,
            151645,
            94,
            151644,
            273,
            105,
            94,
            38,
            229,
            362,
            98,
            1695,
            310,
            1305,
            165,
            128,
            432,
            43,
            44,
            82,
            13,
            151645,
            94,
            151644,
            397,
            251,
            249,
            94,
            151643,
        ]
        expected_formatted_messages = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "Give me a short introduction to LLMs.<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<|endoftext|>"
        )
        _test_tokenize_messages(
            tokenizer,
            messages,
            expected_tokens,
            expected_formatted_messages,
        )

    def test_tool_call(self):
        tokenizer = self.tokenizer()
        messages = [
            Message(role="system", content="a"),
            Message(role="user", content="b"),
            Message(role="assistant", content="test call", ipython=True),
            Message(role="ipython", content="test response"),
            Message(role="assistant", content=""),
        ]
        expected_tokens = [
            151644,
            82,
            88,
            479,
            94,
            64,
            151645,
            94,
            151644,
            273,
            105,
            94,
            65,
            151645,
            94,
            151644,
            397,
            251,
            249,
            94,
            151657,
            94,
            83,
            269,
            107,
            330,
            94,
            151658,
            151645,
            94,
            151644,
            273,
            105,
            94,
            27,
            83,
            1364,
            62,
            237,
            79,
            102,
            182,
            29,
            94,
            83,
            269,
            706,
            102,
            182,
            94,
            1932,
            83,
            1364,
            62,
            237,
            79,
            102,
            182,
            29,
            151645,
            94,
            151644,
            397,
            251,
            249,
            94,
            151643,
        ]
        expected_formatted_messages = (
            "<|im_start|>system\n"
            "a<|im_end|>\n"
            "<|im_start|>user\n"
            "b<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<tool_call>\n"
            "test call\n"
            "</tool_call><|im_end|>\n"
            "<|im_start|>user\n"
            "<tool_response>\n"
            "test response\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
            "<|endoftext|>"
        )
        _test_tokenize_messages(
            tokenizer,
            messages,
            expected_tokens,
            expected_formatted_messages,
        )


def _test_tokenize_messages(
    tokenizer, messages, expected_tokens, expected_formatted_messages
):
    tokens, mask = tokenizer.tokenize_messages(messages)
    assert len(tokens) == len(mask)
    assert expected_tokens == tokens
    formatted_messages = tokenizer.decode(tokens)
    assert expected_formatted_messages == formatted_messages
