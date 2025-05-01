# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torchtune.modules.loss.loss_protocols import RLLinearLoss


class LinearGRPOLoss(nn.Module, RLLinearLoss):
    """Memory efficient GRPO loss that incrementally computes loss for chunks of tokens
    by masking ignored tokens, calculating logits and then applying GRPO loss. Combines
    the linear projection with the GRPO calculation for futher memory savings.
    """

    def __init__(
        self,
        num_output_chunks: int = 8,
        epsilon: float = 0.1,
        kl_coeff: float = 0.1,
        ignore_index: int = -100,
        mask_pre_projection: bool = True,
    ):
        super().__init__()
        """
        Args:
            num_output_chunks (int): Number of chunks to split the output tensor into. Default is 8.
            ignore_index (int): Index to ignore in the target tensor. Default is -100.
            mask_pre_projection (bool): Whether to mask the output tensor before projection, avoiding
                computing it for tokens that will be ignored during CE anyway. Default is True.
        """
        self.num_output_chunks = num_output_chunks
        self.epsilon = epsilon
        self.kl_coeff = kl_coeff
        self.ignore_index = ignore_index
        self.mask_pre_projection = mask_pre_projection

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the compute_grpo_loss function.
        If compiling CE + chunking operation together, memory requirement is higher."""
        self.chunked_grpo_loss = torch.compile(self.chunked_grpo_loss, *args, **kwargs)
        return self

    def chunked_grpo_loss(
        self,
        weight: torch.Tensor,  # [H, Vocab]
        hidden_chunk: torch.Tensor,  # [B*G, chunk_size, H]
        targets_chunk: torch.Tensor,  # [B*G, chunk_size]
        ref_logprobs_chunk: torch.Tensor,  # [B*G, chunk_size]
        advantages: torch.Tensor,  # [B*G]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logits_chunk = F.linear(hidden_chunk, weight)

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
        weight: torch.Tensor,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        padding_masks: Optional[torch.Tensor] = None,  # [B*G, response_length]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            weight (torch.Tensor): Tensor with weights of the model output projection layer. Shape ``[vocab_size, emb_dim]``
            outputs (torch.Tensor): Hidden state of the model, pre projection. Shape ``[bsz, seq_len, emb_dim]``
            targets (torch.Tensor): Labels for the model. Shape ``[bsz, seq_len]``
            ref_logprobs (torch.Tensor): Reference logprobs for KL loss. Shape ``[bsz, seq_len, vocab_size]``
            advantages (torch.Tensor): Advantages for KL loss. Shape ``[bsz, seq_len, vocab_size?]``
            padding_masks (Optional[torch.Tensor]): Mask for padding tokens. Shape ``[bsz, seq_len]``

        Returns:
            Tuple[torch.Tensor, ...]: loss, policy_loss, kl_loss, ratios, clipfrac, pi_logprobs
        """
        # Chunk along sequence dimension
        hidden_chunks = outputs.tensor_split(self.num_output_chunks, dim=1)
        target_chunks = targets.tensor_split(self.num_output_chunks, dim=1)
        ref_logprobs_chunks = ref_logprobs.tensor_split(self.num_output_chunks, dim=1)

        # Default to all-ones mask if padding_masks is None
        if padding_masks is None:
            padding_masks = torch.ones_like(targets, dtype=torch.bool)
        padding_masks_chunks = padding_masks.tensor_split(self.num_output_chunks, dim=1)

        # Initialize accumulators
        batch_size = advantages.numel()
        device = outputs.device

        total_loss_sum = torch.zeros(batch_size, device=device)
        total_policy_sum = torch.zeros(batch_size, device=device)
        total_kl_sum = torch.zeros(batch_size, device=device)
        total_token_count = torch.zeros(batch_size, device=device)
        pi_logprobs_list = []  # Collect pi_logprobs for each chunk

        # Process each chunk
        for chunk_idx in range(self.num_output_chunks):
            (
                per_token_loss_chunk,
                per_token_policy_loss_chunk,
                per_token_kl_chunk,
                pi_logprobs_chunk,
            ) = self.chunked_grpo_loss(
                weight,
                hidden_chunks[chunk_idx],
                target_chunks[chunk_idx],
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
