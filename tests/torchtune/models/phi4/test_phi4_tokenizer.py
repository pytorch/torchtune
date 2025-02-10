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


class TestPhi4MiniTokenizer:
    @pytest.fixture
    def tokenizer(self):
        # Pretrained tiktoken model generated via the script in
        # https://gist.github.com/ebsmothers/54b133dd87db6679b14318545aaa2de4
        return phi4_tokenizer(
            path=str(ASSETS / "tiktoken_small.model"),
        )

    @pytest.fixture
    def expected_tokens(self):
        # fmt: off
        tokens = [100257, 100264, 115, 121, 322, 398, 100265, 10, 1539, 470, 258, 1444, 933, 1940, 511, 446, 100266, 10, 100264,
          477, 273, 100265, 10, 66, 478, 299, 351, 362, 292, 1160, 117, 807, 334, 958, 99, 445, 98, 300, 258, 256, 281,
          107, 46, 411, 114, 561, 258, 1156, 279, 316, 334, 604, 337, 112, 445, 1827, 512, 1080, 116, 300, 262, 1249,
          524, 340, 10, 35, 35, 35, 828, 1160, 117, 807, 1037, 71, 1414, 534, 258, 1759, 511, 355, 285, 875, 550, 102,
          1546, 265, 105, 111, 340, 10, 35, 35, 35, 408, 300, 112, 279, 316, 1037, 100266, 10, 100264, 520, 511, 446,
          100265, 10, 73, 776, 362, 425, 1978, 274, 284, 1528, 319, 995, 505, 944, 874, 903, 1585, 616, 345, 1528, 115,
          284, 1749, 803, 46, 270, 776, 1341, 258, 1279, 641, 563, 275, 469, 573, 284, 944, 320, 526, 962, 425, 913,
          1402, 97, 356, 446, 115, 284, 1229, 1581, 282, 117, 276, 259, 300, 46, 270, 776, 258, 1279, 275, 288, 283,
          262, 739, 1886, 284, 783, 1803, 636, 277, 268, 117, 316, 485, 115, 284, 302, 416, 273, 900, 46, 270, 776, 591,
          630, 346, 531, 476, 505, 768, 1233, 342, 1923, 292, 522, 662, 280, 274, 913, 601, 359, 300, 44, 335, 834, 335,
          531, 476, 505, 604, 264, 509, 1456, 258, 771, 543, 1719, 405, 710, 665, 668, 1280, 46, 100266, 10,
          100265]  # noqa
        # fmt: on
        return tokens

    def test_tokenize_messages(self, tokenizer, expected_tokens):
        messages = [
            Message(role="system", content="You are a helpful assistant", masked=True),
            Message(
                role="user",
                content="Below is an instruction that describes a task. Write a response "
                "that appropriately completes the request.\n\n### Instruction:\nGenerate "
                "a realistic dating profile bio.\n\n### Response:\n",
                masked=True,
            ),
            Message(
                role="assistant",
                content="I'm an outgoing and friendly person who loves spending time with "
                "friends and family. I'm also a big-time foodie and love trying out new "
                "restaurants and different cuisines. I'm a big fan of the arts and enjoy "
                "going to museums and galleries. I'm looking for someone who shares my "
                "interest in exploring new places, as well as someone who appreciates a "
                "good conversation over coffee.",
            ),
        ]
        tokens, mask = tokenizer.tokenize_messages(messages, add_eos=True)

        expected_mask = [True] * 101 + [False] * 131
        assert expected_tokens == tokens
        assert expected_mask == mask

    def test_tokenize_messages_no_system_prompt(self, tokenizer):
        messages = [
            Message(role="system", content="You are a helpful assistant", masked=True),
            Message(
                role="user",
                content="Below is an instruction that describes a task. Write a response "
                "that appropriately completes the request.\n\n### Instruction:\nGenerate "
                "a realistic dating profile bio.\n\n### Response:\n",
                masked=True,
            ),
            Message(
                role="assistant",
                content="I'm an outgoing and friendly person who loves spending time with "
                "friends and family. I'm also a big-time foodie and love trying out new "
                "restaurants and different cuisines. I'm a big fan of the arts and enjoy "
                "going to museums and galleries. I'm looking for someone who shares my "
                "interest in exploring new places, as well as someone who appreciates a "
                "good conversation over coffee.",
            ),
        ]
        tokens, mask = tokenizer.tokenize_messages(
            messages, ignore_system_prompt=True, add_eos=True
        )

        # fmt: off
        expected_tokens = [100257, 100264, 477, 273, 100265, 10, 66, 478, 299, 351, 362, 292, 1160, 117, 807, 334, 958, 99, 445,
                   98, 300, 258, 256, 281, 107, 46, 411, 114, 561, 258, 1156, 279, 316, 334, 604, 337, 112, 445, 1827,
                   512, 1080, 116, 300, 262, 1249, 524, 340, 10, 35, 35, 35, 828, 1160, 117, 807, 1037, 71, 1414, 534,
                   258, 1759, 511, 355, 285, 875, 550, 102, 1546, 265, 105, 111, 340, 10, 35, 35, 35, 408, 300, 112,
                   279, 316, 1037, 100266, 10, 100264, 520, 511, 446, 100265, 10, 73, 776, 362, 425, 1978, 274, 284,
                   1528, 319, 995, 505, 944, 874, 903, 1585, 616, 345, 1528, 115, 284, 1749, 803, 46, 270, 776, 1341,
                   258, 1279, 641, 563, 275, 469, 573, 284, 944, 320, 526, 962, 425, 913, 1402, 97, 356, 446, 115, 284,
                   1229, 1581, 282, 117, 276, 259, 300, 46, 270, 776, 258, 1279, 275, 288, 283, 262, 739, 1886, 284,
                   783, 1803, 636, 277, 268, 117, 316, 485, 115, 284, 302, 416, 273, 900, 46, 270, 776, 591, 630, 346,
                   531, 476, 505, 768, 1233, 342, 1923, 292, 522, 662, 280, 274, 913, 601, 359, 300, 44, 335, 834, 335,
                   531, 476, 505, 604, 264, 509, 1456, 258, 771, 543, 1719, 405, 710, 665, 668, 1280, 46, 100266, 10,
                   100265]  # noqa
        # fmt: on

        expected_mask = [True] * 84 + [False] * 131
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
                content="Below is an instruction that describes a task. Write a response "
                "that appropriately completes the request.\n\n### Instruction:\nGenerate "
                "a realistic dating profile bio.\n\n### Response:\n",
                masked=True,
            ),
            Message(
                role="assistant",
                content="I'm an outgoing and friendly person who loves spending time with "
                "friends and family. I'm also a big-time foodie and love trying out new "
                "restaurants and different cuisines. I'm a big fan of the arts and enjoy "
                "going to museums and galleries. I'm looking for someone who shares my "
                "interest in exploring new places, as well as someone who appreciates a "
                "good conversation over coffee.",
            ),
        ]

        tokens, mask = tokenizer.tokenize_messages(messages, add_eos=False)

        # fmt: off
        expected_tokens = [100257, 100264, 115, 121, 322, 398, 100265, 10, 1539, 470, 258, 1444, 933, 1940, 511, 446, 100266,
        10, 100264, 477, 273, 100265, 10, 66, 478, 299, 351, 362, 292, 1160, 117, 807, 334, 958, 99, 445, 98,
        300, 258, 256, 281, 107, 46, 411, 114, 561, 258, 1156, 279, 316, 334, 604, 337, 112, 445, 1827, 512,
        1080, 116, 300, 262, 1249, 524, 340, 10, 35, 35, 35, 828, 1160, 117, 807, 1037, 71, 1414, 534, 258,
        1759, 511, 355, 285, 875, 550, 102, 1546, 265, 105, 111, 340, 10, 35, 35, 35, 408, 300, 112, 279,
        316, 1037, 100266, 10, 100264, 520, 511, 446, 100265, 10, 73, 776, 362, 425, 1978, 274, 284, 1528,
        319, 995, 505, 944, 874, 903, 1585, 616, 345, 1528, 115, 284, 1749, 803, 46, 270, 776, 1341, 258,
        1279, 641, 563, 275, 469, 573, 284, 944, 320, 526, 962, 425, 913, 1402, 97, 356, 446, 115, 284, 1229,
        1581, 282, 117, 276, 259, 300, 46, 270, 776, 258, 1279, 275, 288, 283, 262, 739, 1886, 284, 783,
        1803, 636, 277, 268, 117, 316, 485, 115, 284, 302, 416, 273, 900, 46, 270, 776, 591, 630, 346, 531,
        476, 505, 768, 1233, 342, 1923, 292, 522, 662, 280, 274, 913, 601, 359, 300, 44, 335, 834, 335, 531,
        476, 505, 604, 264, 509, 1456, 258, 771, 543, 1719, 405, 710, 665, 668, 1280, 46, 100266, 10,
        100265]  # noqa
        # fmt: on

        expected_mask = [True] * 101 + [False] * 130
        # Drop eos token.
        assert expected_tokens[:-1] == tokens
        assert expected_mask == mask
