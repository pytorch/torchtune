# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest
from torchtune.data import Message
from torchtune.models.llama2 import llama2_tokenizer, Llama2ChatTemplate
from collections import Counter

ASSETS = Path(__file__).parent.parent.parent.parent / "assets"


class TestLlama2Tokenizer:
    def tokenizer(self, template: bool = False):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        return llama2_tokenizer(str(ASSETS / "m.model"), prompt_template=Llama2ChatTemplate() if template else None)

    @pytest.fixture
    def messages(self):
        return [
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

    def test_tokenize_messages(self, messages):
        tokenizer = self.tokenizer(template=False)
        tokens, mask = tokenizer.tokenize_messages(messages)
        expected_tokens = {}
        expected_mask = {}
        assert expected_tokens == Counter(tokens)
        assert expected_mask == Counter(mask)

    def test_tokenize_messages_chat_template(self, messages):
        tokenizer = self.tokenizer(template=True)
        tokens, mask = tokenizer.tokenize_messages(messages)
        expected_tokens = {}
        expected_mask = {}
        assert expected_tokens == Counter(tokens)
        assert expected_mask == Counter(mask)
