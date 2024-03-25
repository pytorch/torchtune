# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.data import tokenize_prompt_and_response, truncate_if_necessary
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX


class DummyTokenizer:
    def encode(self, text, **kwargs):
        words = text.split()
        return [len(word) for word in words]

    @property
    def eos_id(self):
        return -1


def test_tokenize_prompt_and_response():
    tokenizer = DummyTokenizer()
    prompt = "Instruction:\nThis is an instruction.\n\nInput:\nThis is an input.\n\nResponse: "
    response = "I always know what I'm doing, do you?"
    prompt_length = 11
    expected_tokenized_prompt = [
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


def test_truncate_if_necessary():
    prompt_tokens = [1, 2, 3, 4, -1]
    label_tokens = [1, 2, 3, 4, -1]
    max_seq_len = 5

    # Test no truncation
    truncated_prompt_tokens, truncated_label_tokens = truncate_if_necessary(
        tokenizer=DummyTokenizer(),
        prompt_tokens=prompt_tokens,
        label_tokens=label_tokens,
        max_seq_len=max_seq_len,
    )
    assert truncated_prompt_tokens == [1, 2, 3, 4, -1]
    assert truncated_label_tokens == [1, 2, 3, 4, -1]

    # Test truncated
    max_seq_len = 4
    truncated_prompt_tokens, truncated_label_tokens = truncate_if_necessary(
        tokenizer=DummyTokenizer(),
        prompt_tokens=prompt_tokens,
        label_tokens=label_tokens,
        max_seq_len=max_seq_len,
    )
    assert truncated_prompt_tokens == [1, 2, 3, -1]
    assert truncated_label_tokens == [1, 2, 3, -1]
