# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tests.common import ASSETS
from torchtune.data import Message
from torchtune.modules.transforms.tokenizers import HuggingFaceModelTokenizer

TOKENIZER_CONFIG_PATH = ASSETS / "tokenizer_config_gemma.json"
GENERATION_CONFIG_PATH = ASSETS / "generation_config_gemma.json"
TOKENIZER_PATH = ASSETS / "tokenizer_gemma_cropped.json"


class TestHuggingFaceModelTokenizer:
    @pytest.fixture
    def model_tokenizer(self):
        return HuggingFaceModelTokenizer(
            tokenizer_json_path=str(TOKENIZER_PATH),
            tokenizer_config_json_path=str(TOKENIZER_CONFIG_PATH),
            generation_config_path=str(GENERATION_CONFIG_PATH),
        )

    @pytest.fixture
    def messages(self):
        return [
            Message(
                role="user",
                content="hello there",
                masked=False,
            ),
            Message(
                role="assistant",
                content="hi",
                masked=False,
            ),
            Message(
                role="user",
                content="whatsup?",
                masked=False,
            ),
        ]

    def test_no_mask(self, model_tokenizer, messages):
        tokens, mask = model_tokenizer.tokenize_messages(messages)

        assert tokens[:-4] == [
            2,
            106,
            1645,
            108,
            17534,
            1104,
            107,
            108,
            106,
            2516,
            108,
            544,
            107,
            108,
            106,
            1645,
            108,
            5049,
            15827,
            235336,
            107,
            108,
        ]
        assert mask[:-4] == [False] * 22

    def test_with_mask(self, model_tokenizer, messages):
        """
        In this test we mask the first message and verify that it does not affect tokens,
        and message mask is changed in the correct way.
        """
        messages[0].masked = True
        tokens, mask = model_tokenizer.tokenize_messages(messages)

        assert tokens[:-4] == [
            2,
            106,
            1645,
            108,
            17534,
            1104,
            107,
            108,
            106,
            2516,
            108,
            544,
            107,
            108,
            106,
            1645,
            108,
            5049,
            15827,
            235336,
            107,
            108,
        ]

        assert mask == [True] * 8 + [False] * 14
