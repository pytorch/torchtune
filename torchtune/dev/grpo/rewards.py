# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from xml.etree import ElementTree as ET

import torch

from torchtune.modules.transforms.tokenizers import ModelTokenizer


def extract_tags(text: str) -> dict[str, list[str]]:
    """
    Parse XML-like tags from text. Returns a dictionary with keys 'think' and 'answer'.
    The values are lists of strings, with each string being the content of a tag.
    """
    xml_string = f"<root>{text}</root>"
    root = ET.fromstring(xml_string)

    return {
        "think": [
            elem.text if elem.text is not None else "" for elem in root.findall("think")
        ],
        "answer": [
            elem.text if elem.text is not None else ""
            for elem in root.findall("answer")
        ],
    }


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
) -> [torch.Tensor, torch.Tensor]:
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
