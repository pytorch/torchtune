# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import numpy as np
from typing import Dict, List, Any, List

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules import Tokenizer

from torchtune.data._types import Message


def truncate(
    tokens: List[Any],
    max_seq_len: int,
    eos_id: Any,
) -> List[Any]:
    tokens_truncated = tokens[:max_seq_len]
    if tokens_truncated[-1] != eos_id:
        tokens_truncated[-1] = eos_id
    return tokens_truncated


def validate_messages(
    messages: List[Message],
) -> None:
    """
    Given a list of messages, ensure that messages form a valid
    back-and-forth conversation. An error will be raised if:
        - There is a system message that's not the first message
        - There are two consecutive user messages
        - An assistant message comes before the first user message
        - The message is empty
        - Messages are shorter than length of 2 (min. one user-assistant turn)

    Args:
        messages (List[Message]): the messages to validate.

    Raises:
        ValueError: If the messages are invalid.
    """
    if len(messages) < 2:
        raise ValueError(
            f"Messages must be at least length 2, but got {len(messages)} messages"
        )

    last_turn = "assistant"
    for i, message in enumerate(messages):
        if message.role == "assistant" and last_turn != "user":
            raise ValueError(
                f"Assistant message before expected user message at index {i} in messages"
            )
        if message.role == "user" and last_turn == "user":
            raise ValueError(
                f"Two consecutive user messages at index {i} and {i - 1} in messages"
            )
        if message.role == "system" and i > 0:
            raise ValueError(
                f"System message at index {i} in messages, but system messages must come first"
            )
        # Assistant messages can be empty because they will not be tokenized and
        # will not contribute to the loss, assuming the entire batch is not full
        # of empty assistant messages. The alpaca dataset is an example of the
        # output assistant message being empty sometimes.
        if not message.content and message.role != "assistant":
            raise ValueError(f"Message at index {i} in messages is empty")
        last_turn = message.role

def build_tokenized_answer(tokenizer, prompt, answer):
    """
    Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
    It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
    Reference:
        https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    """
    full_tokenized = tokenizer.encode(prompt + answer, add_bos=False, add_eos=False)
    prompt_input_ids = tokenizer.encode(prompt, add_bos=False, add_eos=False)

    answer_input_ids = full_tokenized[len(prompt_input_ids) :]

    # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

    # Prepare input tokens for token by token comparison
    full_input_ids = np.array(full_tokenized)

    if len(full_input_ids) != len(full_concat_input_ids):
        raise ValueError("Prompt input ids and answer input ids should have the same length.")

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = len(prompt_input_ids)

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    if prompt_input_ids != full_tokenized[:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized[:response_token_ids_start_idx]

    answer_input_ids = full_tokenized[response_token_ids_start_idx:]

    return dict(
        prompt_input_ids=prompt_input_ids,
        input_ids=answer_input_ids,
    )