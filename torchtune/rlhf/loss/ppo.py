# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchtune import rlhf


class PPOLoss(nn.Module):
    """
    Proximal Policy Optimization (PPO) Loss module.
    This implementation uses the following references:

    https://arxiv.org/abs/1707.06347 eqn. 7

    https://github.com/vwxyzjn/lm-human-preference-details/blob/ccc19538e817e98a60d3253242ac15e2a562cb49/lm_human_preference_details/train_policy_accelerate.py#L719

    https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L68-L75


    Args:
        epsilon (float): clipping range for PPO update.
        value_clip_range (float): clipping range for value function update.
        value_coeff (float): coefficient for the value function loss contribution.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        value_clip_range: float = 0.2,
        value_coeff: float = 0.1,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.value_clip_range = value_clip_range
        self.value_coeff = value_coeff

    def forward(
        self,
        pi_old_logprobs: torch.Tensor,
        pi_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        phi_old_values: torch.Tensor,
        phi_values: torch.Tensor,
        returns: torch.Tensor,
        padding_masks: Optional[torch.Tensor] = None,
        value_padding_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Forward pass of the PPO loss module.

        Args:
            pi_old_logprobs (torch.Tensor): Log probabilities of the old policy.
            pi_logprobs (torch.Tensor): Log probabilities of the current policy.
            advantages (torch.Tensor): Advantage values.
            phi_old_values (torch.Tensor): Value predictions of the old value function.
            phi_values (torch.Tensor): Value predictions of the current value function.
            returns (torch.Tensor): Return values.
            padding_masks (Optional[torch.Tensor]): Padding token masks of the same shape as ``pi_logprobs``,
                where True indicates the corresponding loss values should participage in policy loss calculation.
            value_padding_masks (Optional[torch.Tensor]): Padding token masks of the same shape as ``pi_logprobs``,
                where True indicates the corresponding loss values should participage in value loss calculation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of five tensors:
                - loss: The total PPO loss.
                - policy_loss: The policy function loss.
                - value_loss: The value function loss.
                - ratios: The ratio between the current and old policy probabilities.
                - clipfrac: The fraction of ratios that were clipped.

        """
        ratios = torch.exp(pi_logprobs - pi_old_logprobs)
        clipped_ratios = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon)

        policy_losses_clipped = -advantages * clipped_ratios
        policy_losses_unclipped = -advantages * ratios

        clipfrac = (policy_losses_clipped > policy_losses_unclipped).float()
        clipfrac = (
            clipfrac.mean()
            if padding_masks is None
            else rlhf.masked_mean(clipfrac, padding_masks)
        )

        policy_loss = torch.maximum(policy_losses_clipped, policy_losses_unclipped)
        policy_loss = (
            policy_loss.mean()
            if padding_masks is None
            else rlhf.masked_mean(policy_loss, padding_masks)
        )

        values_clipped = torch.clamp(
            phi_values,
            phi_old_values - self.value_clip_range,
            phi_old_values + self.value_clip_range,
        )
        value_loss = torch.maximum(
            (phi_values - returns) ** 2, (values_clipped - returns) ** 2
        )
        value_loss = (
            0.5 * value_loss.mean()
            if value_padding_masks is None
            else 0.5 * rlhf.masked_mean(value_loss, value_padding_masks)
        )

        loss = policy_loss + (value_loss * self.value_coeff)
        return (
            loss,
            policy_loss.detach(),
            value_loss.detach(),
            ratios.mean().detach(),
            clipfrac.detach(),
        )
