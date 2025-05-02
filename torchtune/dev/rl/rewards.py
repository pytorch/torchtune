# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import List, Tuple
from xml.etree import ElementTree as ET

import math_verify

import torch

from torchtune.modules.transforms.tokenizers import ModelTokenizer


def extract_tags(text: str) -> Tuple[str, str]:
    """
    Parse XML-like tags from text. Returns a dictionary with keys 'think' and 'answer'.
    The values are lists of strings, with each string being the content of a tag.
    """
    think_pattern = r"<think>(.*?)</think>"
    answer_pattern = r"<answer>(.*?)</answer>"
    think_match = re.search(think_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    cot = think_match.group(1).strip() if think_match else ""
    potential_answer = answer_match.group(1).strip() if answer_match else ""
    return cot, potential_answer


def at_least_one_space_between_think_tags(
    cot: str, answer: str, potential_answer: str
) -> Tuple[float, float]:
    """Did the model at least try to think?"""
    if len(cot) > 0:
        return 1.0, 1.0  # (reward, success)
    else:
        return 0.0, 0.0


def math_response_correct(
    cot: str, answer: str, potential_answer: str
) -> Tuple[float, float]:
    """Did it get the right answer?"""
    if potential_answer is None:
        return 0.0, 0.0  # (reward, success)
    gold = math_verify.parse(answer)
    attempt = math_verify.parse(potential_answer)

    if math_verify.verify(gold, attempt):
        return 100.0, 1.0
    if answer in potential_answer:
        return 50.0, 0.0
    if len(potential_answer) > 0:
        return 1.0, 0.0
    return 0.0, 0.0


def batched_rewards(
    tokenizer: ModelTokenizer,
    completions: torch.Tensor,
    answers: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:

    reward_funcs = [
        at_least_one_space_between_think_tags,
        math_response_correct,
    ]
    num_reward_funcs = len(reward_funcs)
    batch_size, grpo_size, _ = completions.shape

    # TODO: should this be bfloat16?
    rewards_tensor = torch.zeros(
        batch_size, grpo_size, num_reward_funcs, dtype=torch.float32, device=device
    )
    successes_tensor = torch.zeros(
        batch_size, grpo_size, num_reward_funcs, dtype=torch.float32, device=device
    )
    metadata = {"func_names": [f.__name__ for f in reward_funcs]}
    for b in range(batch_size):
        for g in range(grpo_size):
            answer = answers[b][g]
            text_completion = tokenizer.decode(completions[b, g].tolist())
            cot, potential_answer = extract_tags(f"<think>{text_completion}")
            for rw_idx, reward_func in enumerate(reward_funcs):
                reward, success = reward_func(cot, answer, potential_answer)
                rewards_tensor[b, g, rw_idx] += reward
                successes_tensor[b, g, rw_idx] += success

    return rewards_tensor, successes_tensor, metadata


def shaped_correctness_reward(answer: str, completion: str) -> tuple[float, float]:
    """
    Reward function for verifiable rewards with some mild shaping.

    Args:
        answer (str): ground-truth answer to the current problem
        completion (str): model's completion, starting immediately after "Assistant: <think>"
    Returns:
        reward: (float) a shaped reward indicating the correct answer and the correct format
        success: (float) a binary measure of success (1 if the answer is correct and correctly formatted, 0 otherwise)
    """
    reward = 0.0
    success = 0.0

    try:
        tags = extract_tags("<think>" + completion.replace("<<", "").replace(">>", ""))
    except ET.ParseError:
        tags = {"think": [], "answer": []}

    if len(tags["answer"]) == 1:
        reward += 5.0

    if len(tags["think"]) == 1:
        reward += 5.0

    if any(attempt == answer for attempt in tags["answer"]):
        # One of the answer tags has the right answer
        reward += 20.0

    if any((answer in attempt) for attempt in tags["answer"]):
        # One of the answer tags contains the right answer (might be e.g. $20 instead of 20)
        reward += 10.0

    if len(tags["answer"]) > 0 and tags["answer"][-1] == answer:
        reward = 100.0
        success = 1

    return reward, success


def batch_shaped_correctness_reward(
    tokenizer: ModelTokenizer, completions: torch.Tensor, answers: list[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Utility function to apply the shaped reward function to a GRPO-style batch of completions."""

    batch_size, grpo_size, *_ = completions.shape
    rewards = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
    successes = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
    # completions :: [B, G, L]
    for b in range(batch_size):
        for g in range(grpo_size):
            text_completion = tokenizer.decode(
                completions[b, g].tolist()
            )  # skips special tokens, stops at eos
            reward, success = shaped_correctness_reward(
                answer=answers[b], completion=text_completion
            )
            rewards[b, g] = reward
            successes[b, g] = success

    return rewards, successes
