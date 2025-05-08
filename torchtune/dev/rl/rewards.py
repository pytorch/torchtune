# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class RewardOutput:
    """
    This class is used to store the reward and other statistics for a given reward function.

    Attributes:
        reward_base_name (str): the base name of the reward function, e.g. "math_correctness" or "formatting"
        total_reward (torch.Tensor): the total reward for the given reward function, shape ``[b]``
        successes (torch.Tensor): the number of successes for the given reward function, shape ``[b]``
        rewards (Optional[Dict[str, torch.Tensor]]): an optional dictionary of sub-rewards for the given reward function,
           which are only used for logging purposes. e.g:
           ``{"soft_format_reward": torch.Tensor, "strict_format_reward": torch.Tensor}``
    """

    reward_base_name: str
    total_reward: torch.Tensor
    successes: torch.Tensor
    rewards: Optional[Dict[str, torch.Tensor]] = field(default_factory=dict)

    def log(self, prefix: str = "") -> Dict[str, float]:
        """
        Logs the reward and other statistics for the given reward function.

        Args:
            prefix (str): an optional prefix to add to the log keys

        Returns:
            A dictionary of the logged statistics.
        Example:
            >>> reward_output = RewardOutput(
                reward_base_name="math_correctness",
                total_reward=torch.tensor([1.0, 2.0, 3.0]),
                successes=torch.tensor([1.0, 0.0, 1.0]),
                rewards={"soft_format_reward": torch.tensor([1.0, 0.0, 1.0]), "strict_format_reward": torch.tensor([1.0, 0.0, 1.0])}
            )
            >>> reward_output.log(prefix="train")
            {
                "train/math_correctness": 2.0,
                "train/math_correctness/successes": 0.6666666666666666,
                "train/math_correctness/soft_format_reward": 1.0,
                "train/math_correctness/strict_format_reward": 1.0
            }
        """
        log_dict = {}
        prefix = (
            f"{prefix}/{self.reward_base_name}" if prefix else self.reward_base_name
        )

        for reward_name, reward in self.rewards.items():
            log_dict[f"{prefix}/{reward_name}"] = reward.mean().item()

        log_dict[f"{prefix}"] = self.total_reward.mean().item()
        log_dict[f"{prefix}/successes"] = self.successes.mean().item()
        return log_dict


class Reward(ABC):
    """
    This is an abstract base class for rewards which are used in GRPO.
    """

    @abstractmethod
    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: List[str],
        answers: List[str],
    ) -> RewardOutput:
        """
        This method is called to compute the reward for a given completion and answer.

        Args:
            completion_ids (torch.Tensor): the token ids of the completion, shape ``[b, seq_len]``
            completions (List[str]): the completions, shape ``[b, seq_len]``
            answers (List[str]): the answers, shape ``[b, seq_len]``

        Returns:
            A ``RewardOutput`` object containing the total reward to be used in advantage estimation,
                alongside additional metadata useful for logging.
        """
        pass


class FormattedMathCorrectnessReward(Reward):
    """
    This reward encourages the model to correctly answer a math problem, and requires
    the model to repond in an XML-style format to extract answers.

    Args:
        answer_tag (str): the tag for the answer section. The answer will be extracted from <{answer_tag}>{answer}</{answer_tag}>
        positive_reward (float): the reward provided for correctly formatted completions
        negative_reward (float): the reward provided for incorrectly formatted completions
    """

    def __init__(
        self, answer_tag: str, positive_reward: float, negative_reward: float = 0.0
    ):
        self.answer_tag = answer_tag
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: List[str],
        answers: List[str],
    ) -> RewardOutput:
        rewards = []
        import math_verify

        for completion, answer in zip(completions, answers):
            gold = math_verify.parse(answer)
            attempt = math_verify.parse(completion)
            if math_verify.verify(gold, attempt):
                reward = self.positive_reward
            elif answer in completion:
                reward = self.positive_reward / 2
            else:
                reward = self.negative_reward
            rewards.append(reward)

        rewards = torch.tensor(rewards)
        return RewardOutput(
            reward_base_name="math_correctness",
            total_reward=rewards,
            successes=(rewards == self.positive_reward).float(),
        )


class ThinkingAnswerFormattingReward(Reward):
    """
    This reward encourages the model to respond in a reasoning-style format. It applies
    both a soft and strict formatting reward.

    The "soft" formatting reward rewards the model for using the tags, even if the tags do not
    have newlines.

    The "strict" formatting reward rewards the model for using the tags, and having newlines.

    Taken from https://github.com/huggingface/open-r1/blob/06bdd503341f5375bf93c3720df13f8588d47712/src/open_r1/rewards.py

    Args:
        think_tag (str): the tag for the think section. The tag will be XML-formatted as <{think_tag}>...</{think_tag}>
        answer_tag (str): the tag for the answer section. The tag will be XML-formatted as <{answer_tag}>...</{answer_tag}>
        positive_reward (float): the reward provided for correctly formatted completions
        negative_reward (float): the reward provided for incorrectly formatted completions
    """

    def __init__(
        self,
        think_tag: str,
        answer_tag: str,
        positive_reward: float,
        negative_reward: float = 0.0,
    ):
        self.think_tag = think_tag
        self.answer_tag = answer_tag
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

    def __call__(
        self,
        completion_ids: torch.Tensor,
        completions: List[str],
        answers: List[str],
    ) -> RewardOutput:
        # soft format reward pattern
        think_pattern = rf"<{self.think_tag}>.*?</{self.think_tag}>"
        answer_pattern = rf"<{self.answer_tag}>.*?</{self.answer_tag}>"

        # strict format reward pattern
        strict_pattern = rf"^<{self.think_tag}>\n.*?\n</{self.think_tag}>\n<{self.answer_tag}>\n.*?\n</{self.answer_tag}>\n$"
        soft_format_rewards = []
        strict_format_rewards = []
        for completion in completions:
            strict_format_rewards.append(
                self.positive_reward
                if re.match(strict_pattern, completion, re.DOTALL | re.MULTILINE)
                else self.negative_reward
            )

            think_matches = re.findall(think_pattern, completion, re.DOTALL)
            answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
            if len(think_matches) == 1 and len(answer_matches) == 1:
                think_index = completion.find(think_matches[0])
                answer_index = completion.find(answer_matches[0])
                if think_index < answer_index:
                    soft_format_rewards.append(self.positive_reward)
                    continue
            soft_format_rewards.append(self.negative_reward)

        soft_format_rewards = torch.tensor(soft_format_rewards)
        strict_format_rewards = torch.tensor(strict_format_rewards)
        rewards = soft_format_rewards + strict_format_rewards
        successes = (rewards >= self.positive_reward).float()
        return RewardOutput(
            reward_base_name="formatting",
            total_reward=rewards,
            rewards={
                "soft_format_reward": soft_format_rewards,
                "strict_format_reward": strict_format_rewards,
            },
            successes=successes,
        )
