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


from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRPOWithChunkedOutputLoss(nn.Module):
    """
    GRPO loss with chunked output to reduce memory usage by upcasting one chunk at a time.

    Args:
        num_output_chunks (int): Number of chunks to split the sequence into. If 0, expects non-chunked input.
        epsilon (float): Clipping range for GRPO update (unused here).
        kl_coeff (float): KL divergence coefficient (beta).
    """

    def __init__(
        self, num_output_chunks: int = 8, epsilon: float = 0.1, kl_coeff: float = 0.1
    ):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.epsilon = epsilon
        self.kl_coeff = kl_coeff

    def compute_per_token_quantities(
        self,
        pi_logits_chunk: torch.Tensor,  # [B*G, chunk_size, V]
        targets_chunk: torch.Tensor,  # [B*G, chunk_size]
        ref_logprobs_chunk: torch.Tensor,  # [B*G, chunk_size]
        advantages: torch.Tensor,  # [B*G]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # CE
        pi_logits_flat = pi_logits_chunk.reshape(-1, pi_logits_chunk.size(-1))
        targets_flat = targets_chunk.reshape(-1)
        pi_logprobs_chunk = -F.cross_entropy(
            pi_logits_flat.float(), targets_flat, reduction="none"
        )
        pi_logprobs_chunk = pi_logprobs_chunk.view_as(targets_chunk)

        # Detach
        pi_logprobs_detached = pi_logprobs_chunk.detach()
        ref_logprobs_detached = ref_logprobs_chunk.detach()

        # KL term
        per_token_kl = (
            torch.exp(ref_logprobs_detached - pi_logprobs_chunk)
            - (ref_logprobs_detached - pi_logprobs_chunk)
            - 1
        )

        # Policy term
        per_token_policy_loss = (
            torch.exp(pi_logprobs_chunk - pi_logprobs_detached) * advantages[:, None]
        )

        # Total per-token loss
        per_token_loss = -(per_token_policy_loss - self.kl_coeff * per_token_kl)

        return per_token_loss, per_token_policy_loss, per_token_kl, pi_logprobs_chunk

    def forward(
        self,
        pi_logits: (
            torch.Tensor | List[torch.Tensor]
        ),  # [B*G, response_length, V] or List[[B*G, chunk_size, V]]
        targets: torch.Tensor,  # [B*G, response_length]
        ref_logprobs: torch.Tensor,  # [B*G, response_length]
        advantages: torch.Tensor,  # [B*G]
        padding_masks: Optional[torch.Tensor] = None,  # [B*G, response_length]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Compute GRPO loss over chunked or full logits.

        Args:
            pi_logits (torch.Tensor | List[torch.Tensor]): Logits of the policy model. If a list, each element is a chunk of logits.
            targets (torch.Tensor): Targets for the policy model.
            ref_logprobs (torch.Tensor): Log probabilities of the reference model.
            advantages (torch.Tensor): Advantage values.
            padding_masks (Optional[torch.Tensor]): Padding token masks where True indicates tokens to include in loss calculation.

        Returns:
            Tuple of (loss, policy_loss, kl_loss, ratios, clipfrac, pi_logprobs).
        """
        # Handle chunked or non-chunked pi_logits
        if isinstance(pi_logits, torch.Tensor):
            pi_logits = [pi_logits]
        num_chunks = len(pi_logits)

        # Chunk sequence tensors
        targets_chunks = targets.chunk(num_chunks, dim=1)
        ref_logprobs_chunks = ref_logprobs.chunk(num_chunks, dim=1)

        # Default to all-ones mask if padding_masks is None
        if padding_masks is None:
            padding_masks = torch.ones_like(targets, dtype=torch.bool)
        padding_masks_chunks = padding_masks.chunk(num_chunks, dim=1)

        # Initialize accumulators
        batch_size = advantages.numel()
        device = pi_logits[0].device
        total_loss_sum = torch.zeros(batch_size, device=device)
        total_policy_sum = torch.zeros(batch_size, device=device)
        total_kl_sum = torch.zeros(batch_size, device=device)
        total_token_count = torch.zeros(batch_size, device=device)
        pi_logprobs_list = []  # Collect pi_logprobs for each chunk

        # Process each chunk
        for chunk_idx in range(num_chunks):
            (
                per_token_loss_chunk,
                per_token_policy_loss_chunk,
                per_token_kl_chunk,
                pi_logprobs_chunk,
            ) = self.compute_per_token_quantities(
                pi_logits[chunk_idx],
                targets_chunks[chunk_idx],
                ref_logprobs_chunks[chunk_idx],
                advantages,
            )

            # Accumulate with padding mask applied
            padding_masks_chunk = padding_masks_chunks[chunk_idx]
            total_loss_sum += (per_token_loss_chunk * padding_masks_chunk).sum(dim=1)
            with torch.no_grad():
                total_policy_sum += (
                    per_token_policy_loss_chunk * padding_masks_chunk
                ).sum(dim=1)
                total_kl_sum += (per_token_kl_chunk * padding_masks_chunk).sum(dim=1)
                total_token_count += padding_masks_chunk.sum(dim=1)

            # Store pi_logprobs for this chunk
            pi_logprobs_list.append(pi_logprobs_chunk)

        # Concatenate pi_logprobs across all chunks
        pi_logprobs = torch.cat(pi_logprobs_list, dim=1)  # [B*G, response_length]

        # Compute mean losses per sequence, then average over batch
        total_token_count = total_token_count.clamp(min=1e-9)
        loss = (total_loss_sum / total_token_count).mean()
        with torch.no_grad():
            policy_loss = (total_policy_sum / total_token_count).mean()
            kl_loss = (total_kl_sum / total_token_count).mean()

        # Dummy values for unused metrics
        ratios = torch.tensor(1.0, device=device)
        clipfrac = torch.tensor(0.0, device=device)

        return loss, policy_loss, kl_loss, ratios, clipfrac, pi_logprobs
