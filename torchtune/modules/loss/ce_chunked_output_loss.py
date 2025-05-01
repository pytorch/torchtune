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

        # Apply ratio if provided
        if ratio is not None:
            loss = loss * ratio.squeeze(0)

        # Sum the losses
        return loss.sum()

    def _calculate_importance_ratio(
        self,
        new_log_ps: torch.Tensor,
        ref_log_ps: torch.Tensor,
        labels: torch.Tensor,
        epsilon_low: float = 0.0,
        epsilon_high: float = float("inf"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the importance ratio for the new and reference log probabilities.

        Args:
            new_log_ps: Log probabilities from the current model
            ref_log_ps: Log probabilities from the reference model
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

            # Use the valid indices for gathering
            new_selected = torch.gather(
                new_log_ps, dim=-1, index=valid_indices.unsqueeze(-1)
            ).squeeze(-1)

            old_selected = torch.gather(
                ref_log_ps, dim=-1, index=valid_indices.unsqueeze(-1)
            ).squeeze(-1)

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
            ref_logprob_chunks = [
                r_chunk.reshape(-1, r_chunk.size(-1)) for r_chunk in ref_logprobs
            ]

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

    def compute_entropy(self, logits: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the entropy of the model's output probabilities.

        Args:
            logits (List[Tensor]): List of chunked logits, each of shape
                (batch_size, num_tokens / num_chunks, vocab_size)

        Returns:
            Tensor: Scalar entropy value.
        """
        entropies = []
        for logit_chunk in logits:
            # shape: (batch_size * chunk_size, vocab_size)
            log_probs = torch.nn.functional.log_softmax(
                logit_chunk.reshape(-1, logit_chunk.size(-1)), dim=-1
            )
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(
                dim=-1
            )  # shape: (batch_size * chunk_size,)
            entropies.append(entropy)

        all_entropies = torch.cat(entropies)  # flatten
        return all_entropies.mean()
