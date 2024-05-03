# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest
from torchtune.data._types import Message
from torchtune.modules.tokenizers import SentencePieceTokenizer

ASSETS = Path(__file__).parent.parent.parent.parent / "assets"


class TestSentencePieceTokenizer:
    @pytest.fixture
    def tokenizer(self):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        return SentencePieceTokenizer(str(ASSETS / "m.model"))

    def test_encode(self, tokenizer):
        assert tokenizer.encode("Hello world!") == [
            tokenizer.bos_id,
            12,
            1803,
            1024,
            103,
            tokenizer.eos_id,
        ]
        assert tokenizer.encode("Hello world!", add_eos=False) == [
            tokenizer.bos_id,
            12,
            1803,
            1024,
            103,
        ]
        assert tokenizer.encode("Hello world!", add_bos=False) == [
            12,
            1803,
            1024,
            103,
            tokenizer.eos_id,
        ]
        assert tokenizer.encode("Hello world!", add_eos=False, add_bos=False) == [
            12,
            1803,
            1024,
            103,
        ]

    def test_decode(self, tokenizer):
        assert tokenizer.decode([1, 12, 1803, 1024, 103, 2]) == "Hello world!"

    def test_token_ids(self, tokenizer):
        assert tokenizer.eos_id == 2
        assert tokenizer.pad_id == -1
        assert tokenizer.bos_id == 1

    def test_tokenizer_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 2000

    def test_encode_without_leading_whitespace(self, tokenizer):
        s1 = "Hello"
        s2 = "I'm an outgoing and friendly person."
        s1_tokens = tokenizer.encode(s1, add_bos=False, add_eos=False)
        s2_tokens = tokenizer.encode(s2, add_bos=False, add_eos=False)
        # Set prefix="pre" since "\n" is not in the test tokenizer's vocab
        s2_tokens_no_whitespace = tokenizer.encode(
            s2, add_bos=False, add_eos=False, trim_leading_whitespace=True, prefix="pre"
        )
        s1s2_tokens = tokenizer.encode(s1 + s2, add_bos=False, add_eos=False)
        assert (s1_tokens + s2_tokens) != s1s2_tokens
        assert (s1_tokens + s2_tokens_no_whitespace) == s1s2_tokens

    def test_tokenize_messages(self, tokenizer):
        messages = [
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
        tokens, mask = tokenizer.tokenize_messages(messages)
        expected_tokens = [
            1,
            323,
            418,
            202,
            31,
            128,
            15,
            120,
            47,
            88,
            584,
            23,
            1665,
            182,
            9,
            434,
            295,
            85,
            4,
            780,
            47,
            636,
            9,
            1094,
            213,
            23,
            9,
            69,
            69,
            164,
            1153,
            299,
            35,
            961,
            132,
            237,
            7,
            5,
            761,
            4,
            12,
            0,
            313,
            120,
            47,
            88,
            584,
            166,
            493,
            171,
            54,
            299,
            9,
            906,
            244,
            19,
            186,
            767,
            303,
            671,
            92,
            209,
            24,
            190,
            52,
            38,
            4,
            12,
            0,
            1243,
            7,
            69,
            135,
            213,
            166,
            6,
            21,
            45,
            128,
            71,
            58,
            38,
            14,
            10,
            652,
            35,
            462,
            101,
            1306,
            7,
            341,
            171,
            20,
            14,
            127,
            26,
            652,
            7,
            10,
            1268,
            4,
            6,
            21,
            45,
            591,
            9,
            566,
            22,
            994,
            913,
            38,
            20,
            52,
            24,
            10,
            1306,
            734,
            14,
            71,
            365,
            1382,
            7,
            10,
            801,
            105,
            88,
            244,
            985,
            7,
            4,
            6,
            21,
            45,
            9,
            566,
            126,
            180,
            11,
            5,
            1137,
            7,
            10,
            1089,
            151,
            8,
            1156,
            213,
            342,
            7,
            10,
            384,
            104,
            54,
            470,
            4,
            6,
            21,
            45,
            287,
            14,
            33,
            125,
            135,
            24,
            101,
            512,
            66,
            7,
            28,
            822,
            15,
            542,
            69,
            59,
            110,
            14,
            365,
            229,
            7,
            3,
            36,
            267,
            36,
            125,
            135,
            24,
            101,
            1503,
            182,
            9,
            222,
            1661,
            191,
            332,
            92,
            92,
            24,
            24,
            4,
            2,
        ]
        expected_mask = [True] * 75 + [False] * 125
        assert expected_tokens == tokens
        assert expected_mask == mask
