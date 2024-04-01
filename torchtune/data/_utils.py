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

def tokenize_preference_sample(
        tokenizer: Tokenizer,
        prompt: str,
        chosen: str ,
        rejected: str,
        max_length: int = 512,
        max_prompt_length: int = 128,
        truncation_mode: str = "keep_end",
) -> Dict:
    batch = {} 
    if not isinstance(prompt, str):
        raise ValueError(f"prompt should be an str but got {type(prompt)}")
    prompt_tokens = tokenizer.encode(prompt, add_bos=False, add_eos=False)
    prompt_tokens = {"prompt_input_ids":  prompt_tokens}

    if not isinstance(chosen, str):
        raise ValueError(f"chosen should be an str but got {type(chosen)}")
    chosen_tokens = build_tokenized_answer(tokenizer, prompt, chosen)

    if not isinstance(rejected, str):
        raise ValueError(f"rejected should be an str but got {type(rejected)}")
    rejected_tokens = build_tokenized_answer(tokenizer, prompt, rejected)

    # Last prompt token might get merged by tokenizer and
    # it should not be included for generation if that happens
    prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

    chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
    rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
    prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

    for k, v in prompt_tokens.items():
        prompt_tokens[k] = v[:prompt_len_input_ids]

    # Make sure prompts only have one different token at most an
    # and length only differs by 1 at most
    num_diff_tokens = sum(
        [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
    )
    num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
    if num_diff_tokens > 1 or num_diff_len > 1:
        raise ValueError(
            "Chosen and rejected prompt_input_ids might only differ on the "
            "last token due to tokenizer merge ops."
        )

    # add BOS token to head of prompt
    prompt_tokens["prompt_input_ids"] = [tokenizer.bos_id] + prompt_tokens["prompt_input_ids"]
    chosen_tokens["prompt_input_ids"] = [tokenizer.bos_id] + chosen_tokens["prompt_input_ids"]
    rejected_tokens["prompt_input_ids"] = [tokenizer.bos_id] + rejected_tokens["prompt_input_ids"]

    # add EOS token to end of answer
    chosen_tokens["input_ids"].append(tokenizer.eos_id)
    rejected_tokens["input_ids"].append(tokenizer.eos_id)

    longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

    # if combined sequence is too long, truncate the prompt
    for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
        if len(answer_tokens["prompt_input_ids"]) + longer_response_length > max_length:
            if truncation_mode == "keep_start":
                for k in ["prompt_input_ids"]:
                    answer_tokens[k] = answer_tokens[k][: max_prompt_length]
            elif truncation_mode == "keep_end":
                for k in ["prompt_input_ids"]:
                    answer_tokens[k] = answer_tokens[k][-max_prompt_length :]
        

    # if that's still too long, truncate the response
    for answer_tokens in [chosen_tokens, rejected_tokens]:
        if len(answer_tokens["prompt_input_ids"]) + longer_response_length > max_length:
            for k in ["input_ids"]:
                answer_tokens[k] = answer_tokens[k][: max_length - max_prompt_length]

    # Create labels
    chosen_sequence_tokens = {
        k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids"]
    }
    rejected_sequence_tokens = {
        k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids"]
    }
    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
        CROSS_ENTROPY_IGNORE_IDX
    ] * len(chosen_tokens["prompt_input_ids"])
    rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
        CROSS_ENTROPY_IGNORE_IDX
    ] * len(rejected_tokens["prompt_input_ids"])

    batch = dict(
        chosen_input_ids = chosen_sequence_tokens['input_ids'],
        chosen_labels = chosen_sequence_tokens['labels'],
        rejected_input_ids = rejected_sequence_tokens['input_ids'],
        rejected_labels = rejected_sequence_tokens['labels']
    )
    return batch 