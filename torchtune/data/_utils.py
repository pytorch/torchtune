# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import List, Tuple

from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules import Tokenizer


def tokenize_prompt_and_response(
    tokenizer: Tokenizer,
    prompt: str,
    response: str,
    train_on_input: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Tokenize a prompt and response pair.

    Args:
        tokenizer (Tokenizer): The tokenizer to use.
        prompt (str): The prompt to tokenize.
        response (str): The response to tokenize.
        train_on_input (bool): Whether to train on prompt or mask it out. Default
            is False.

    Returns:
        The tokenized prompt and response.
    """
    prompt_with_response = prompt + response
    encoded_prompt = tokenizer.encode(text=prompt, add_bos=True, add_eos=False)
    encoded_prompt_with_response = tokenizer.encode(
        text=prompt_with_response, add_bos=True, add_eos=True
    )
    labels = copy.deepcopy(encoded_prompt_with_response)

    if not train_on_input:
        labels[: len(encoded_prompt)] = [CROSS_ENTROPY_IGNORE_IDX] * len(encoded_prompt)

    assert len(encoded_prompt_with_response) == len(labels)

    return encoded_prompt_with_response, labels


def truncate(
    tokenizer: Tokenizer,
    prompt_tokens: List[int],
    label_tokens: List[int],
    max_seq_len: int,
) -> Tuple[List[int], List[int]]:
    """
    Truncate a prompt and label pair if longer than max sequence length.

    Args:
        tokenizer (Tokenizer): The tokenizer to use.
        prompt_tokens (List[int]): The prompt + response tokens.
        label_tokens (List[int]): The label tokens.
        max_seq_len (int): The maximum sequence length.

    Returns:
        The truncated prompt and label.
    """
    prompt_tokens_truncated = prompt_tokens[:max_seq_len]
    label_tokens_truncated = label_tokens[:max_seq_len]
    if prompt_tokens_truncated[-1] != tokenizer.eos_id:
        prompt_tokens_truncated[-1] = tokenizer.eos_id
    if label_tokens_truncated[-1] != tokenizer.eos_id:
        label_tokens_truncated[-1] = tokenizer.eos_id

    return prompt_tokens_truncated, label_tokens_truncated
