# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.test_utils import DummyTokenizer
from torchtune.data import tokenize_prompt_and_response, truncate
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX


def test_tokenize_prompt_and_response():
    tokenizer = DummyTokenizer()
    prompt = "Instruction:\nThis is an instruction.\n\nInput:\nThis is an input.\n\nResponse: "
    response = "I always know what I'm doing, do you?"
    prompt_length = 12
    expected_tokenized_prompt = [
        -1,
        12,
        4,
        2,
        2,
        12,
        6,
        4,
        2,
        2,
        6,
        9,
        1,
        6,
        4,
        4,
        3,
        6,
        2,
        4,
    ]
    expected_tokenized_label = [CROSS_ENTROPY_IGNORE_IDX] * prompt_length + [
        1,
        6,
        4,
        4,
        3,
        6,
        2,
        4,
        -1,
    ]

    tokenized_prompt, tokenized_label = tokenize_prompt_and_response(
        tokenizer, prompt, response
    )
    assert tokenized_prompt == expected_tokenized_prompt
    assert tokenized_label == expected_tokenized_label

    tokenized_prompt, tokenized_label = tokenize_prompt_and_response(
        tokenizer, prompt, response, train_on_input=True
    )
    assert tokenized_prompt == expected_tokenized_prompt
    assert tokenized_label == expected_tokenized_prompt


def test_truncate():
    prompt_tokens = [1, 2, 3, 4, -1]
    label_tokens = [1, 2, 3, 4, -1]

    # Test no truncation
    truncated_prompt_tokens, truncated_label_tokens = truncate(
        tokenizer=DummyTokenizer(),
        prompt_tokens=prompt_tokens,
        label_tokens=label_tokens,
        max_seq_len=5,
    )
    assert truncated_prompt_tokens == prompt_tokens
    assert truncated_label_tokens == label_tokens

    # Test truncated
    truncated_prompt_tokens, truncated_label_tokens = truncate(
        tokenizer=DummyTokenizer(),
        prompt_tokens=prompt_tokens,
        label_tokens=label_tokens,
        max_seq_len=4,
    )
    assert truncated_prompt_tokens == [1, 2, 3, -1]
    assert truncated_label_tokens == [1, 2, 3, -1]
