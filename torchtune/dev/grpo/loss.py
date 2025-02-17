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
    Group Relative Policy Optimization (GRPO) Loss module.
    Introduced by https://arxiv.org/abs/2402.03300, popularized by https://arxiv.org/abs/2501.12948.

    This loss implementation follows the usual formulation of GRPO with clipped ratios of token-wise logprobs.
    Currently not validated to perform well.

    Args:
        epsilon (float): clipping range for GRPO update.
        kl_coeff (float): KL divergence coefficient (also known as beta).
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        kl_coeff: float = 0.1,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.kl_coeff = kl_coeff

    def forward(
        self,
        pi_old_logprobs: torch.Tensor,  # [B x G, L]
        pi_logprobs: torch.Tensor,  # [B x G, L]
        ref_logprobs: torch.Tensor,  # [B x G, L]
        advantages: torch.Tensor,  # [B x G]
        padding_masks: Optional[torch.Tensor] = None,  # [B x G, L]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GRPO loss module.

        Args:
            pi_old_logprobs (torch.Tensor): Log probabilities of the old policy. Shape: [batch_size * num_groups, seq_len]
            pi_logprobs (torch.Tensor): Log probabilities of the current policy. Shape: [batch_size * num_groups, seq_len]
            ref_logprobs (torch.Tensor): Log probabilities of the reference model. Shape: [batch_size * num_groups, seq_len]
            advantages (torch.Tensor): Advantage values. Shape: [batch_size * num_groups]
            padding_masks (Optional[torch.Tensor]): Padding token masks where True indicates tokens to include in loss calculation.
                Shape: [batch_size * num_groups, seq_len]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - loss: Total GRPO loss (policy loss + KL penalty)
                - policy_loss: Clipped policy loss
                - kl_loss: KL divergence loss between policy and reference model
                - ratios: Mean ratio between current and old policy probabilities
                - clipfrac: Fraction of clipped policy ratios
        """

        ratios = torch.exp(pi_logprobs - pi_old_logprobs)  # [B x G, L]
        clipped_ratios = torch.clamp(
            ratios, 1.0 - self.epsilon, 1.0 + self.epsilon
        )  # [B x G, L]

        advantages = advantages[:, None]  # [B x G, 1]

        policy_losses_clipped = advantages * clipped_ratios  # [B x G, L]
        policy_losses_unclipped = advantages * ratios  # [B x G, L]

        clipfrac = (
            policy_losses_clipped < policy_losses_unclipped
        ).float()  # [B x G, L]
        clipfrac = rlhf.masked_mean(clipfrac, padding_masks)  # scalar

        policy_loss = torch.minimum(
            policy_losses_clipped, policy_losses_unclipped
        )  # [B x G, L]
        policy_loss = rlhf.masked_mean(policy_loss, padding_masks)

        kl_loss = (
            torch.exp(ref_logprobs - pi_logprobs) - (ref_logprobs - pi_logprobs) - 1
        )  # [B x G]
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
    Group Relative Policy Optimization (GRPO) Loss module.
    Introduced by https://arxiv.org/abs/2402.03300, popularized by https://arxiv.org/abs/2501.12948.

    This loss implementation follows the usual formulation of GRPO with clipped ratios of full completion logprobs.
    Currently not validated to perform well.

    Args:
        epsilon (float): clipping range for GRPO update.
        kl_coeff (float): KL divergence coefficient (also known as beta).
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        kl_coeff: float = 0.1,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.kl_coeff = kl_coeff

    def forward(
        self,
        pi_old_logprobs: torch.Tensor,  # [B x G, L]
        pi_logprobs: torch.Tensor,  # [B x G, L]
        ref_logprobs: torch.Tensor,  # [B x G, L]
        advantages: torch.Tensor,  # [B x G]
        padding_masks: Optional[torch.Tensor] = None,  # [B x G, L]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GRPO loss module.

        Args:
            pi_old_logprobs (torch.Tensor): Log probabilities of the old policy. Shape: [batch_size * num_groups, seq_len]
            pi_logprobs (torch.Tensor): Log probabilities of the current policy. Shape: [batch_size * num_groups, seq_len]
            ref_logprobs (torch.Tensor): Log probabilities of the reference model. Shape: [batch_size * num_groups, seq_len]
            advantages (torch.Tensor): Advantage values. Shape: [batch_size * num_groups]
            padding_masks (Optional[torch.Tensor]): Padding token masks where True indicates tokens to include in loss calculation.
                Shape: [batch_size * num_groups, seq_len]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - loss: Total GRPO loss (policy loss + KL penalty)
                - policy_loss: Clipped policy loss
                - kl_loss: KL divergence loss between policy and reference model
                - ratios: Mean ratio between current and old policy probabilities
                - clipfrac: Fraction of clipped policy ratios
        """

        pi_old_logprobs = masked_sum(pi_old_logprobs, padding_masks)  # [B x G]
        pi_logprobs = masked_sum(pi_logprobs, padding_masks)  # [B x G]
        ref_logprobs = masked_sum(ref_logprobs, padding_masks)  # [B x G]

        ratios = torch.exp(pi_logprobs - pi_old_logprobs)  # [B x G]
        clipped_ratios = torch.clamp(
            ratios, 1.0 - self.epsilon, 1.0 + self.epsilon
        )  # [B x G]

        policy_losses_clipped = advantages * clipped_ratios  # [B x G]
        policy_losses_unclipped = advantages * ratios  # [B x G]

        clipfrac = (policy_losses_clipped < policy_losses_unclipped).float()  # [B x G]
        clipfrac = clipfrac.mean()  # scalar, only for logging

        policy_loss = torch.minimum(
            policy_losses_clipped, policy_losses_unclipped
        )  # [B x G]
        policy_loss = policy_loss.mean()  # scalar

        kl_loss = (
            torch.exp(ref_logprobs - pi_logprobs) - (ref_logprobs - pi_logprobs) - 1
        )  # [B x G]
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
    Group Relative Policy Optimization (GRPO) Loss module.
    Introduced by https://arxiv.org/abs/2402.03300, popularized by https://arxiv.org/abs/2501.12948.

    This loss implementation is based on TRL's implementation of GRPO,
     which only takes a single gradient step per batch, trivializing some parts of the computation.
     This empirically seems to perform well.

    Args:
        epsilon (float): clipping range for GRPO update.
        kl_coeff (float): KL divergence coefficient (also known as beta).
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        kl_coeff: float = 0.1,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.kl_coeff = kl_coeff

    def forward(
        self,
        pi_old_logprobs: torch.Tensor,  # [B x G, L]
        pi_logprobs: torch.Tensor,  # [B x G, L]
        ref_logprobs: torch.Tensor,  # [B x G, L]
        advantages: torch.Tensor,  # [B x G]
        padding_masks: Optional[torch.Tensor] = None,  # [B x G, L]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GRPO loss module.

        Args:
            pi_old_logprobs (torch.Tensor): *UNUSED* Log probabilities of the old policy.
                Shape: [batch_size * num_groups, seq_len]
            pi_logprobs (torch.Tensor): Log probabilities of the current policy.
                Shape: [batch_size * num_groups, seq_len]
            ref_logprobs (torch.Tensor): *UNUSED* Log probabilities of the reference model.
                Shape: [batch_size * num_groups, seq_len]
            advantages (torch.Tensor): Advantage values.
                Shape: [batch_size * num_groups]
            padding_masks (Optional[torch.Tensor]): Padding token masks where True indicates tokens to include in loss calculation.
                Shape: [batch_size * num_groups, seq_len]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - loss: Total GRPO loss (policy loss + KL penalty)
                - policy_loss: Clipped policy loss
                - kl_loss: KL divergence loss between policy and reference model
                - ratios: Mean ratio between current and old policy probabilities
                - clipfrac: Fraction of clipped policy ratios
        """

        # [B x G, L]
        per_token_kl = (
            torch.exp(ref_logprobs.detach() - pi_logprobs)
            - (ref_logprobs.detach() - pi_logprobs)
            - 1
        )

        advantages = advantages[:, None]  # [B x G, 1]

        per_token_policy_loss = (
            torch.exp(pi_logprobs - pi_logprobs.detach()) * advantages
        )

        per_token_loss = -(per_token_policy_loss - self.kl_coeff * per_token_kl)

        loss = rlhf.masked_mean(per_token_loss, padding_masks, dim=1).mean()

        policy_loss = (
            rlhf.masked_mean(per_token_policy_loss, padding_masks, dim=1)
            .mean()
            .detach()
        )
        kl_loss = rlhf.masked_mean(per_token_kl, padding_masks, dim=1).mean().detach()

        return (  # This loss doesn't track clipfrac and ratios
            loss,
            policy_loss,
            kl_loss,
            torch.tensor(1.0),
            torch.tensor(0.0),
        )
