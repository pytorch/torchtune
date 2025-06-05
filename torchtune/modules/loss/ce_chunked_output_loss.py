# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from torchtune import config, modules, training, utils

log = utils.get_logger("DEBUG")


class CEWithChunkedOutputLoss(torch.nn.Module):
    """
    Cross-entropy with chunked outputs that saves memory by only upcasting one chunk at a time.

    Whenever the model is trained with bf16, before running CE, we have to upcast
    it to fp32 for better accuracy and stability. When upcasting happens, the memory usage doubles.
    Models like llama3 have large vocabulary size and, therefore, have a large output
    tensor of shape ``(bsz, num_tokens, vocab_size)``. If we chunk on the token level, you can still compute
    the cross entropy normally, but upcasting only one chunk at a time saves considerable memory.

    The CE and upcasting have to be compiled together for better performance.
    When using this class, we recommend using :func:`torch.compile` only on the method ``compute_cross_entropy``.
    The gains from chunking won't be realized if you compile the entire class.

    For more details, please refer to: https://github.com/pytorch/torchtune/pull/1390
    """

    def __init__(self, num_output_chunks: int = 8, ignore_index: int = -100, **kwargs):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index

    def compute_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ratio: torch.Tensor = None,
        normalize: bool = True,
        epsilon = 1e-10
    ) -> torch.Tensor:
        """
        Upcast logits to fp32 and compute cross entropy loss.

        Args:
            logits: Input logits tensor
            labels: Ground truth labels
            ratio: Optional per-token importance weights or advantages
            normalize: Whether to normalize the loss
        """
        # Standard cross entropy loss

        loss = F.cross_entropy(
            logits.float(), labels, ignore_index=self.ignore_index, reduction="none"
        )
        self.epsilon = epsilon
        # Apply ratio if provided
        if ratio is not None:
            loss = loss * ratio.squeeze(0)

        # Sum the losses
        return loss.sum()

    def _calculate_importance_ratio(
        self,
        new_log_ps: torch.Tensor,
        old_log_ps: torch.Tensor,  # This is now the pre-gathered value
        labels: torch.Tensor,
        epsilon_low: float = 0.0,
        epsilon_high: float = float("inf"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the importance ratio for the new and reference log probabilities.

        Args:
            new_log_ps: Log probabilities from the current model
            old_log_ps: Precomputed log probabilities from the reference model
            labels: Token indices for selecting the correct log probabilities
            epsilon_low: Lower bound for importance ratio clipping
            epsilon_high: Upper bound for importance ratio clipping

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of (unclipped_ratio, clipped_ratio)
        """
        with torch.no_grad():
            # Create a mask for tokens that are the ignore index
            ignore_mask = labels == self.ignore_index

            # For valid calculation, clamp indices to be within vocab range
            vocab_size = new_log_ps.size(-1)
            valid_indices = labels.clamp(0, vocab_size - 1)

            # Use the valid indices for gathering only current model log probs
            new_selected = torch.gather(
                new_log_ps, dim=-1, index=valid_indices.unsqueeze(-1)
            ).squeeze(-1)

            # old_log_ps is already the selected value
            old_selected = old_log_ps

            # Calculate the importance ratio (unclipped)
            importance_ratio = torch.exp(new_selected - old_selected)

            # Set importance ratio to 1.0 for ignored tokens (won't affect loss)
            importance_ratio = torch.where(
                ignore_mask, torch.ones_like(importance_ratio), importance_ratio
            )

            # Get the clipped version of importance ratio
            clipped_importance_ratio = torch.clamp(
                importance_ratio, min=epsilon_low, max=epsilon_high
            )

        return importance_ratio, clipped_importance_ratio

    def forward(
        self,
        logits: Union[torch.Tensor, List[torch.Tensor]],
        labels: torch.Tensor,
        ref_logprobs: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        reward: Optional[torch.Tensor] = None,
        epsilon_low: float = 0.0,
        epsilon_high: float = float("inf"),
    ) -> torch.Tensor:
        """
        Args:
            logits (Union[torch.Tensor, List[torch.Tensor]]): Either a single logits tensor or list of chunked logits,
                where each chunk has shape ``(batch_size, num_tokens / num_output_chunks, vocab_size)``.
            labels (torch.Tensor): Ground truth labels of shape ``(batch_size, num_tokens)``.
            ref_logits (Optional[Union[torch.Tensor, List[torch.Tensor]]]): Reference model logits for importance sampling.
            reward (Optional[torch.Tensor]): Reward tensor for scaling importance ratio.
            epsilon_low (float): Lower bound for importance ratio clipping.
            epsilon_high (float): Upper bound for importance ratio clipping.

        Returns:
            torch.Tensor: Cross entropy loss of shape (1,).
        """
        # Normalization factor
        total_elements = (labels != self.ignore_index).sum()

        # Chunk and reshape labels
        labels_chunks = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]

        # Reshape logits chunks
        logits_chunks = [
            logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits
        ]

        # Preprocess reward
        has_per_token_reward = (
            reward is not None
            and not isinstance(reward, (int, float))
            and reward.numel() > 1
            and reward.shape == labels.shape
        )

        reward_chunks = None
        scalar_reward = None

        if has_per_token_reward:
            # Process per-token rewards
            reward_chunks = [
                r_chunk.reshape(-1)
                for r_chunk in reward.chunk(self.num_output_chunks, dim=1)
            ]
        elif reward is not None:
            # Convert to scalar reward
            scalar_reward = reward

        # Process reference logprobs for importance sampling
        ratio_chunks = None
        if ref_logprobs is not None:
            ref_logprob_chunks = [r_chunk.reshape(-1) for r_chunk in ref_logprobs]

            # Precompute importance ratios
            ratio_chunks = []
            for i, (logits_chunk, ref_logprob_chunk, labels_chunk) in enumerate(
                zip(logits_chunks, ref_logprob_chunks, labels_chunks)
            ):
                # Get log probabilities
                curr_log_ps = F.log_softmax(logits_chunk.float(), dim=-1)

                # Calculate importance ratio (now returns unclipped and clipped)
                unclipped_ratio, clipped_ratio = self._calculate_importance_ratio(
                    curr_log_ps,
                    ref_logprob_chunk,
                    labels_chunk,
                    epsilon_low,
                    epsilon_high,
                )

                # Apply reward to ratios if provided (as PPO-style advantages)
                if has_per_token_reward:
                    reward_chunk = reward_chunks[i]
                    # PPO-style: min(r * A, clip(r) * A)
                    unclipped_value = unclipped_ratio * reward_chunk
                    clipped_value = clipped_ratio * reward_chunk
                    # Choose the minimum for pessimistic bound (PPO style)
                    chunk_ratio = torch.min(unclipped_value, clipped_value)
                elif reward is not None:
                    # Scalar reward case - same principle
                    unclipped_value = unclipped_ratio * scalar_reward
                    clipped_value = clipped_ratio * scalar_reward
                    chunk_ratio = torch.min(unclipped_value, clipped_value)
                else:
                    # No reward provided, use clipped ratio directly
                    chunk_ratio = clipped_ratio

                ratio_chunks.append(chunk_ratio)

        # Compute loss chunk by chunk
        total_loss = 0.0
        for i, (logits_chunk, labels_chunk) in enumerate(
            zip(logits_chunks, labels_chunks)
        ):
            if ratio_chunks is not None:
                # Case 1: Using importance sampling
                chunk_loss = self.compute_cross_entropy(
                    logits_chunk, labels_chunk, ratio_chunks[i]
                )
            elif reward is not None:
                # Case 2: No importance sampling but we have reward
                base_loss = self.compute_cross_entropy(logits_chunk, labels_chunk)

                if has_per_token_reward:
                    chunk_loss = base_loss * reward_chunks[i].mean()
                else:
                    chunk_loss = base_loss * scalar_reward
            else:
                # Case 3: Standard cross-entropy
                chunk_loss = self.compute_cross_entropy(logits_chunk, labels_chunk)

            total_loss += chunk_loss

        # Normalize the loss
        return total_loss / total_elements

    def compute_entropy(self, logits: List[torch.Tensor], labels: torch.Tensor,ent_weight: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Memory-optimized entropy calculation."""
        logits_chunks = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits]
        labels_chunks = [label_chunk.reshape(-1) for label_chunk in labels.chunk(self.num_output_chunks, dim=1)]
        
        # Initialize empty tensors for more efficient operation
        full_token_entropy_mean = 0
        per_token_entropy_sum = 0
        full_token_entropy_sum = 0
        per_token_entropy_mean = 0
        chunk_count = 0
        valid_token_count = 0  # Track valid tokens for proper normalization
        
        # Process chunks one by one
        for i, (logits_chunk, labels_chunk) in enumerate(zip(logits_chunks, labels_chunks)):
            # Create mask for valid tokens (not ignore_index)
            # valid_mask = torch.ones_like(labels_chunk, dtype=torch.bool)
            valid_mask = labels_chunk != self.ignore_index
            valid_tokens_in_chunk = valid_mask.sum()
            
            # Skip chunk if no valid tokens
            if valid_tokens_in_chunk == 0:
                continue
                
            chunk_count += 1
            valid_token_count += valid_tokens_in_chunk
            
            # Calculate entropy components
            log_probs = F.log_softmax(logits_chunk, dim=-1)
            vocab_size = log_probs.size(-1)
            valid_indices = labels_chunk.clamp(0, vocab_size - 1)
            gathered_log_probs = torch.gather(log_probs, dim=-1, index=valid_indices.unsqueeze(-1)).squeeze(-1)
            gathered_probs = gathered_log_probs.exp() + self.epsilon
            
            # Per-token metrics (always detached) - apply mask
            with torch.no_grad():
                per_token_ent = -gathered_probs * gathered_log_probs
                # Apply mask and sum only valid tokens
                per_token_ent_masked = per_token_ent * valid_mask
                per_token_entropy_sum += per_token_ent_masked.sum().detach()
                # Mean only over valid tokens in this chunk
                per_token_entropy_mean += (per_token_ent_masked.sum() / valid_tokens_in_chunk).detach()
            
            # Full token entropy calculation
            if ent_weight > 0:
                ungathered_probs = log_probs.exp() + self.epsilon
                full_token_ent = -ungathered_probs * log_probs
                # Apply mask across vocab dimension by expanding valid_mask
                valid_mask_expanded = valid_mask.unsqueeze(-1)  # Shape: [seq_len, 1]
                full_token_ent_masked = full_token_ent * valid_mask_expanded
                # Only keep gradients for mean computation which is used in loss
                full_token_entropy_mean += (full_token_ent_masked.sum() / valid_tokens_in_chunk).detach() / len(logits_chunks)
            else:
                with torch.no_grad():
                    ungathered_probs = log_probs.exp() + self.epsilon
                    full_token_ent = -ungathered_probs * log_probs
                    # Apply mask
                    valid_mask_expanded = valid_mask.unsqueeze(-1)
                    full_token_ent_masked = full_token_ent * valid_mask_expanded
                    full_token_entropy_mean += (full_token_ent_masked.sum() / valid_tokens_in_chunk) / len(logits_chunks)
            
            with torch.no_grad():
                full_token_entropy_sum += full_token_ent_masked.sum().detach()
            
            # Clean up this chunk's intermediate tensors
            del log_probs, valid_indices, gathered_log_probs, gathered_probs, ungathered_probs
            del full_token_ent, per_token_ent, full_token_ent_masked, per_token_ent_masked
        
        # Normalize means over all valid tokens across chunks
        if chunk_count > 0:
            per_token_entropy_mean = per_token_entropy_mean / chunk_count

        return per_token_entropy_sum, full_token_entropy_sum, per_token_entropy_mean, full_token_entropy_mean    
