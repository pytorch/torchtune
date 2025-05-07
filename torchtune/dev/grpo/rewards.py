# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Dict

import torch

from torchtune.modules.transforms.tokenizers import ModelTokenizer


def extract_tags(text: str) -> Dict:
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
    return {
        "think": [
            cot,
        ],
        "answer": [
            potential_answer,
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


# class ShapedCorrectnessReward(Transform):
#     def __init__(self, tokenizer):
#         super().__init__()
#         self.tokenizer = tokenizer

#     def _step(
#         self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
#     ) -> TensorDictBase:
#         # Get the completion
#         responses = next_tensordict["responses"]  # batch_size, grpo_size, L
#         answers = next_tensordict["answers"]  # batch_size, grpo_size
#         if responses.ndim  == 3:
#             batch_size, grpo_size, _ = responses.shape
#         # decode
#         text_completion = self.tokenizer.decode(
#             responses.flatten(0, 1).tolist()
#         )
#         # Decomposed reward
#         tds = [self.single_shaped_correctness_reward(answer, compl) for answer, compl in zip(answers.view(-1), text_completion)]
#         tds = torch.stack(tds)
#         if responses.ndim  == 3:
#             tds = tds.reshape(batch_size, grpo_size)
#         tds = tds.apply(lambda t: t.unsqueeze(-1))
#         return tds

#     def transform_reward_spec(self, reward_spec: Composite) -> Composite:
#         shape = reward_spec.shape + (1,)
#         reward_spec.update(Composite(
#             reward_answer=Unbounded(shape),
#             reward_think=Unbounded(shape),
#             reward_right=Unbounded(shape),
#             reward_contained=Unbounded(shape),
#             reward=Unbounded(shape),
#             success=Unbounded(shape, dtype=torch.bool),
#         ))
#         return reward_spec

#     @classmethod
#     def single_shaped_correctness_reward(cls, answer: str, completion: str) -> tuple[float, float]:
#         """
#         Reward function for verifiable rewards with some mild shaping.

#         Args:
#             answer (str): ground-truth answer to the current problem
#             completion (str): model's completion, starting immediately after "Assistant: <think>"
#         Returns:
#             reward: (float) a shaped reward indicating the correct answer and the correct format
#             success: (float) a binary measure of success (1 if the answer is correct and correctly formatted, 0 otherwise)
#         """

#         try:
#             tags = extract_tags("<think>" + completion.replace("<<", "").replace(">>", ""))
#         except ET.ParseError:
#             tags = {"think": [], "answer": []}

#         reward_answer = 5.0 * (len(tags["answer"]) == 1)

#         reward_think = 5.0 * (len(tags["think"]) == 1)

#         # One of the answer tags has the right answer
#         reward_right = 20.0 * (any(attempt == answer for attempt in tags["answer"]))

#         # One of the answer tags contains the right answer (might be e.g. $20 instead of 20)
#         reward_contained = 10.0 * (any((answer in attempt) for attempt in tags["answer"]))

#         success = len(tags["answer"]) > 0 and tags["answer"][-1] == answer
#         # Compose the rewards
#         reward = 100.0 * float(success) + (reward_answer + reward_think + reward_contained + reward_right) * (1- float(success))

#         rewards = TensorDict(
#             reward_answer=reward_answer,
#             reward_think=reward_think,
#             reward_right=reward_right,
#             reward_contained=reward_contained,
#             reward=reward,
#             success=success,
#         )
#         return rewards
