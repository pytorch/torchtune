# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchtune import rlhf
from torchtune.rlhf import masked_sum


class GRPOLoss(nn.Module):
    """
    Proximal Policy Optimization (PPO) Loss module.
    This implementation uses the following references:

    https://arxiv.org/abs/1707.06347 eqn. 7

    https://github.com/vwxyzjn/lm-human-preference-details/blob/ccc19538e817e98a60d3253242ac15e2a562cb49/lm_human_preference_details/train_policy_accelerate.py#L719

    https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L68-L75


    Args:
        epsilon (float): clipping range for PPO update.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        kl_coeff: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.kl_coeff = kl_coeff
        self.ignore_index = ignore_index


    def forward(
        self,
        pi_old_logprobs: torch.Tensor,  # [B x G, L]
        pi_logprobs: torch.Tensor,  # [B x G, L]
        ref_logprobs: torch.Tensor,  # [B x G, L]
        advantages: torch.Tensor,  # [B x G]
        padding_masks: Optional[torch.Tensor] = None,  # [B x G, L]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Forward pass of the PPO loss module.

        Args:
            pi_old_logprobs (torch.Tensor): Log probabilities of the old policy.
            pi_logprobs (torch.Tensor): Log probabilities of the current policy.
            advantages (torch.Tensor): Advantage values.
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

        ratios = torch.exp(pi_logprobs - pi_old_logprobs)  # [B x G, L]
        clipped_ratios = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon)  # [B x G, L]


        advantages = advantages[:, None]  # [B x G, 1]

        policy_losses_clipped = advantages * clipped_ratios  # [B x G, L]
        policy_losses_unclipped = advantages * ratios  # [B x G, L]

        clipfrac = (policy_losses_clipped < policy_losses_unclipped).float()  # [B x G, L]
        clipfrac = rlhf.masked_mean(clipfrac, padding_masks)  # scalar

        policy_loss = torch.minimum(policy_losses_clipped, policy_losses_unclipped)  # [B x G, L]
        policy_loss = rlhf.masked_mean(policy_loss, padding_masks)

        kl_loss = (torch.exp(ref_logprobs - pi_logprobs) -
                   (ref_logprobs - pi_logprobs) - 1)  # [B x G]
        kl_loss = rlhf.masked_mean(kl_loss, padding_masks)

        loss = -(policy_loss - self.kl_coeff * kl_loss)

        return (
            loss,
            policy_loss.detach(),
            kl_loss.detach(),
            ratios.mean().detach(),
            clipfrac.detach(),
        )


class GRPOCompletionLoss(nn.Module):
    """
    Proximal Policy Optimization (PPO) Loss module.
    This implementation uses the following references:

    https://arxiv.org/abs/1707.06347 eqn. 7

    https://github.com/vwxyzjn/lm-human-preference-details/blob/ccc19538e817e98a60d3253242ac15e2a562cb49/lm_human_preference_details/train_policy_accelerate.py#L719

    https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L68-L75


    Args:
        epsilon (float): clipping range for PPO update.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        kl_coeff: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.kl_coeff = kl_coeff
        self.ignore_index = ignore_index


    def forward(
        self,
        pi_old_logprobs: torch.Tensor,  # [B x G, L]
        pi_logprobs: torch.Tensor,  # [B x G, L]
        ref_logprobs: torch.Tensor,  # [B x G, L]
        advantages: torch.Tensor,  # [B x G]
        padding_masks: Optional[torch.Tensor] = None,  # [B x G, L]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Forward pass of the PPO loss module.

        Args:
            pi_old_logprobs (torch.Tensor): Log probabilities of the old policy.
            pi_logprobs (torch.Tensor): Log probabilities of the current policy.
            advantages (torch.Tensor): Advantage values.
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

        pi_old_logprobs = masked_sum(pi_old_logprobs, padding_masks)  # [B x G]
        pi_logprobs = masked_sum(pi_logprobs, padding_masks)  # [B x G]
        ref_logprobs = masked_sum(ref_logprobs, padding_masks)  # [B x G]

        ratios = torch.exp(pi_logprobs - pi_old_logprobs)  # [B x G]
        clipped_ratios = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon)  # [B x G]

        policy_losses_clipped = advantages * clipped_ratios  # [B x G]
        policy_losses_unclipped = advantages * ratios  # [B x G]

        clipfrac = (policy_losses_clipped < policy_losses_unclipped).float()  # [B x G]
        clipfrac = clipfrac.mean()  # scalar, only for logging

        policy_loss = torch.minimum(policy_losses_clipped, policy_losses_unclipped)  # [B x G]
        policy_loss = policy_loss.mean()  # scalar

        kl_loss = (torch.exp(ref_logprobs - pi_logprobs) -
                   (ref_logprobs - pi_logprobs) - 1)  # [B x G]
        kl_loss = rlhf.masked_mean(kl_loss, padding_masks)

        loss = -(policy_loss - self.kl_coeff * kl_loss)

        return (
            loss,
            policy_loss.detach(),
            kl_loss.detach(),
            ratios.mean().detach(),
            clipfrac.detach(),
        )


class GRPOSimpleLoss(nn.Module):
    """
    Proximal Policy Optimization (PPO) Loss module.
    This implementation uses the following references:

    https://arxiv.org/abs/1707.06347 eqn. 7

    https://github.com/vwxyzjn/lm-human-preference-details/blob/ccc19538e817e98a60d3253242ac15e2a562cb49/lm_human_preference_details/train_policy_accelerate.py#L719

    https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L68-L75


    Args:
        epsilon (float): clipping range for PPO update.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        kl_coeff: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.kl_coeff = kl_coeff
        self.ignore_index = ignore_index


    def forward(
        self,
        pi_old_logprobs: torch.Tensor,  # [B x G, L]
        pi_logprobs: torch.Tensor,  # [B x G, L]
        ref_logprobs: torch.Tensor,  # [B x G, L]
        advantages: torch.Tensor,  # [B x G]
        padding_masks: Optional[torch.Tensor] = None,  # [B x G, L]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Forward pass of the PPO loss module.

        Args:
            pi_old_logprobs (torch.Tensor): Log probabilities of the old policy.
            pi_logprobs (torch.Tensor): Log probabilities of the current policy.
            advantages (torch.Tensor): Advantage values.
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

        # pi_old_logprobs = masked_sum(pi_old_logprobs, padding_masks)  # [B x G]
        # pi_logprobs = masked_sum(pi_logprobs, padding_masks)  # [B x G]
        # ref_logprobs = masked_sum(ref_logprobs, padding_masks)  # [B x G]

        # [B x G, L]
        per_token_kl = torch.exp(pi_logprobs.detach() - pi_logprobs) - (pi_logprobs.detach() - pi_logprobs) - 1

        advantages = advantages[:, None]  # [B x G, 1]

        per_token_policy_loss = torch.exp(pi_logprobs - pi_logprobs.detach()) * advantages


        per_token_loss = -(per_token_policy_loss - self.kl_coeff * per_token_kl)


        loss = rlhf.masked_mean(per_token_loss, padding_masks, dim=1).mean()

        policy_loss = rlhf.masked_mean(per_token_policy_loss, padding_masks, dim=1).mean().detach()
        kl_loss = rlhf.masked_mean(per_token_kl, padding_masks, dim=1).mean().detach()

        return (  # TODO: readd tracking
            loss,
            policy_loss,
            kl_loss,
            torch.tensor(1.),
            torch.tensor(0.),
        )