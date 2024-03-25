# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest
from torchtune.data import AlpacaInstructTemplate
from torchtune.datasets._common import CROSS_ENTROPY_IGNORE_IDX

from torchtune.datasets._instruct import _get_template
from torchtune.datasets._utils import tokenize_prompt_and_response


class DummyTokenizer:
    def encode(self, text, **kwargs):
        words = text.split()
        return [len(word) for word in words]


def test_tokenize_prompt_and_response():
    tokenizer = DummyTokenizer()
    prompt = "Instruction:\nThis is an instruction.\n\nInput:\nThis is an input.\n\nResponse: "
    response = "I always know what I'm doing, do you?"
    prompt_length = 13
    expected_tokenized_prompt = [
        12, 4, 2, 2, 12, 10, 6, 4, 2, 2, 6, 10, 9, 1, 6, 4, 4, 3, 6, 2, 4,
    ]
    expected_tokenized_label = [CROSS_ENTROPY_IGNORE_IDX] * prompt_length + [1, 6, 4, 4, 3, 6, 2, 4]

    tokenized_prompt, tokenized_label = tokenize_prompt_and_response(tokenizer, prompt, response)
    assert tokenized_prompt == expected_tokenized_prompt
    assert tokenized_label == expected_tokenized_label

    tokenized_prompt, tokenized_label = tokenize_prompt_and_response(tokenizer, prompt, response, train_on_input=True)
    assert tokenized_prompt == expected_tokenized_prompt
    assert tokenized_label == expected_tokenized_prompt
