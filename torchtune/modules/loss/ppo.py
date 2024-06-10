# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn as nn


class PPOLoss(nn.Module):
    """
    Proximal Policy Optimization (PPO) Loss module: https://arxiv.org/abs/1707.06347.

    Using:
        https://github.com/vwxyzjn/lm-human-preference-details/blob/ccc19538e817e98a60d3253242ac15e2a562cb49/lm_human_preference_details/train_policy_accelerate.py#L719
        https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L68-L75

    as references.
    Args:
        gamma (float): Discount factor.
        lmbda (float): lambda parameter for GAE.
        epsilon (float): clipping parameter for PPO.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        lmbda: float = 0.95,
        epsilon: float = 1e-5,
        value_clip_range: float = 0.2,
        value_coeff: float = 0.1,
    ):
        super().__init__()
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.value_clip_range = value_clip_range
        self.value_coeff = value_coeff

    def forward(
        self,
        pi_old_logprobs: torch.Tensor,
        pi_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the PPO loss module.

        Args:
            pi_old_logprobs (torch.Tensor): Log probabilities of the old policy.
            pi_logprobs (torch.Tensor): Log probabilities of the current policy.
            advantage (torch.Tensor): Advantage values.
            values (torch.Tensor): Value predictions.
            returns (torch.Tensor): Return values.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - loss: The total PPO loss.
                - policy_loss: The policy function loss.
                - value_loss: The value function loss.
        """
        ratio = torch.exp(pi_logprobs - pi_old_logprobs)
        clipped_ratios = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)

        policy_losses = -advantages * clipped_ratios
        # taking max instead of min to minimise loss
        policy_loss = torch.max(policy_losses, -advantages * ratio).mean()

        values_clipped = torch.clamp(
            values, values - self.value_clip_range, values + self.value_clip_range
        )
        value_loss = (
            0.5
            * (
                torch.max((values - returns) ** 2, (values_clipped - returns) ** 2)
            ).mean()
        )

        loss = policy_loss + value_loss * self.value_coeff
        return loss, policy_loss, value_loss * self.value_coeff
