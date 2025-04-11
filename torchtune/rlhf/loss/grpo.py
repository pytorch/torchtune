
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
        a_positive: float = 0.1,
        b_positive: float = 0.1,
        a_negative: float = 0.1,
        b_negative: float = 0.1,
        min_: bool = True,
        ignore_index: int = -100
    ):
        super().__init__()
        self.epsilon = epsilon
        self.kl_coeff = kl_coeff
        self.a_positive = a_positive
        self.b_positive = b_positive
        self.a_negative = a_negative
        self.b_negative = b_negative
        self.min_=min_
        self.ignore_index=ignore_index

    def forward(
        self,
        type_: torch.Tensor,
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
        # clipped_ratios = torch.clamp(
        #     ratios, 1.0 - self.epsilon, 1.0 + self.epsilon
        # )  # [B x G, L]
        if type_ == 1:  # Positive trajectory
            clipped_ratios = torch.clamp(ratios, min=self.a_positive, max=self.b_positive)
        else:  # Negative trajectory
            clipped_ratios = torch.clamp(ratios, min=self.a_negative, max=self.b_negative)

        advantages = advantages[:, None]  # [B x G, 1]

        policy_losses_clipped = advantages * clipped_ratios  # [B x G, L]
        policy_losses_unclipped = advantages * ratios  # [B x G, L]

        clipfrac = (
            policy_losses_clipped < policy_losses_unclipped
        ).float()  # [B x G, L]
        clipfrac = rlhf.masked_mean(clipfrac, padding_masks)  # scalar

        if self.min_:

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
        kl_coeff: float = 0.1,
        a_positive: float = 0.1,
        b_positive: float = 0.1,
        a_negative: float = 0.1,
        b_negative: float = 0.1,
        change_to_reinforce: bool = False,
        take_min: bool = True,
        take_clipped: bool = True,
        ignore_index: int = -100

    ):
        super().__init__()
        self.kl_coeff = kl_coeff
        self.a_positive = a_positive
        self.b_positive = b_positive
        self.a_negative = a_negative
        self.b_negative = b_negative
        self.change_to_reinforce=change_to_reinforce
        self.take_min=take_min
        self.take_clipped=take_clipped
        self.ignore_index=ignore_index

    def forward(
        self,
        pi_old_logprobs: torch.Tensor,  # [B x G, L]
        pi_logprobs: torch.Tensor,  # [B x G, L]
        ref_logprobs: torch.Tensor,  # [B x G, L]
        advantages: torch.Tensor,  # [B x G]
        padding_masks: Optional[torch.Tensor] = None,  # [B x G, L],
        type_: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GRPO loss module with REINFORCE fallback.
        """
        pi_old_logprobs = masked_sum(pi_old_logprobs, padding_masks)  # [B x G]
        pi_logprobs = masked_sum(pi_logprobs, padding_masks)  # [B x G]
        ref_logprobs = masked_sum(ref_logprobs, padding_masks)  # [B x G]

        if self.change_to_reinforce == False:
            # GRPO implementation
            ratios = torch.exp(pi_logprobs - pi_old_logprobs)  # [B x G]
            if type_ == 1:  # Positive trajectory
                clipped_ratios = torch.clamp(ratios, min=self.a_positive, max=self.b_positive)
            else:  # Negative trajectory
                clipped_ratios = torch.clamp(ratios, min=self.a_negative, max=self.b_negative)

            policy_losses_clipped = advantages * clipped_ratios  # [B x G]
            policy_losses_unclipped = advantages * ratios  # [B x G]

            clipfrac = (policy_losses_clipped < policy_losses_unclipped).float()  # [B x G]
            clipfrac = clipfrac.mean()  # scalar, only for logging
            if self.take_min:
                policy_loss = torch.minimum(
                    policy_losses_clipped, policy_losses_unclipped
                )  # [B x G]
            elif self.take_clipped:
                policy_loss = policy_losses_clipped
            else:
                policy_loss = policy_losses_unclipped

            policy_loss = policy_loss.mean()  # scalar

            kl_loss = (
                torch.exp(ref_logprobs - pi_logprobs) - (ref_logprobs - pi_logprobs) - 1
            )  # [B x G]
            kl_loss = rlhf.masked_mean(kl_loss, padding_masks)

            loss = -(policy_loss - self.kl_coeff * kl_loss)
        else:
            # REINFORCE implementation
            # In REINFORCE, we directly use log probabilities multiplied by advantages
            policy_loss = (pi_logprobs * advantages).mean()  # Not negated yet
            
            # KL penalty calculation remains the same
            kl_loss = (
                torch.exp(ref_logprobs - pi_logprobs) - (ref_logprobs - pi_logprobs) - 1
            )  # [B x G]
            kl_loss = rlhf.masked_mean(kl_loss, padding_masks)
            
            # Total loss with KL regularization - using the same formula as GRPO
            loss = -(policy_loss)
            
            # For consistency with return values
            ratios = torch.exp(pi_logprobs - pi_old_logprobs)
            clipfrac = torch.zeros_like(loss.detach())

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
    


class GRPOTaperLoss(nn.Module):
    """
    Group Relative Policy Optimization (GRPO) Loss module with Taper Function.
    
    This implementation follows the formulation:
    J_TOPR(π) = ∑_{τ∈T+} μ(τ)ρ(π(τ)/μ(τ), a+, b+)R(τ) + ∑_{τ∈T-} μ(τ)ρ(π(τ)/μ(τ), a-, b-)R(τ)
    
    Optimized for batch size of 1 and single binary trajectory type (positive=1, negative=0).
    
    Args:
        a_positive (float): Lower bound parameter for positive trajectories
        b_positive (float): Upper bound parameter for positive trajectories
        a_negative (float): Lower bound parameter for negative trajectories
        b_negative (float): Upper bound parameter for negative trajectories
        kl_coeff (float): KL divergence coefficient (beta)
    """

    def __init__(
        self,
        a_positive: float = 0.1,
        b_positive: float = 10.0,
        a_negative: float = 0.1,
        b_negative: float = 10.0,
        kl_coeff: float = 0.1,
        ignore_index: int = -100
    ):
        super().__init__()
        self.a_positive = a_positive
        self.b_positive = b_positive
        self.a_negative = a_negative
        self.b_negative = b_negative
        self.kl_coeff = kl_coeff
        self.ignore_index=ignore_index

    def taper_function(self, x: torch.Tensor, a: float, b: float) -> torch.Tensor:
        """
        Implements the taper function ρ(x,a,b) as defined in the equation.
        
        Args:
            x: Input tensor (typically the policy ratio π(τ)/μ(τ))
            a: Lower bound parameter
            b: Upper bound parameter
            
        Returns:
            Tapered values according to the taper function
        """
        result = x.clone()  # Start with original values (for the "otherwise" case)
        
        # For values below a
        if x < a:
            result = a * (1 + torch.log(x / a + 1e-10))
        elif x > b:
            result = b * (1 + torch.log(x / b + 1e-10))
        
        return result

    def forward(
        self,
        pi_old_logprobs: torch.Tensor,  # [1, L]
        pi_logprobs: torch.Tensor,      # [1, L]
        ref_logprobs: torch.Tensor,     # [1, L]
        advantages: torch.Tensor,       # [1]
        padding_masks: Optional[torch.Tensor] = None,  # [1, L]
        type_: int = None,              # Single value: 1=positive, 0=negative
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GRPO loss module with taper function.
        Assumes batch size of 1 and a single binary trajectory type.
        """
        # Sum log probabilities across sequence dimension (masked)
        pi_old_logprobs = masked_sum(pi_old_logprobs, padding_masks)  # [1]
        pi_logprobs = masked_sum(pi_logprobs, padding_masks)          # [1]
        ref_logprobs = masked_sum(ref_logprobs, padding_masks)        # [1]
        
        # Calculate policy ratios: π(τ)/μ(τ)
        ratios = torch.exp(pi_logprobs - pi_old_logprobs)  # [1]
        
        # Apply taper function based on trajectory type
        if type_ == 1:  # Positive trajectory
            tapered_ratios = self.taper_function(ratios, self.a_positive, self.b_positive)
            # Calculate clipfrac for metrics - percentage of values outside [a,b]
            clipfrac = ((ratios < self.a_positive) | (ratios > self.b_positive)).float().mean()
        else:  # Negative trajectory (type_ == 0)
            tapered_ratios = self.taper_function(ratios, self.a_negative, self.b_negative)
            clipfrac = ((ratios < self.a_negative) | (ratios > self.b_negative)).float().mean()
        
        # Calculate policy loss with tapered ratios
        policy_loss = (tapered_ratios * advantages).mean()
        
        # Calculate KL divergence loss
        kl_loss = (
            torch.exp(ref_logprobs - pi_logprobs) - (ref_logprobs - pi_logprobs) - 1
        )  # [1]
        kl_loss = rlhf.masked_mean(kl_loss, padding_masks)
        
        # Total loss with KL penalty
        loss = -(policy_loss - self.kl_coeff * kl_loss)
        
        return (
            loss,
            policy_loss.detach(),
            kl_loss.detach(),
            ratios.mean().detach(),
            clipfrac.detach(),
        )
