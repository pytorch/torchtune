# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn


class ChunkedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss that chunks hidden states and computes loss incrementally,
    masking ignored tokens before projection for efficiency."""

    def __init__(self, num_output_chunks: int = 8, ignore_index: int = -100):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index

    def compute_cross_entropy(
        self,
        weight: torch.Tensor,
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy on a single chunk, only for non-ignored tokens.

        Args:
            weight (torch.Tensor): [vocab_size, embed_dim]
            hidden_chunk (torch.Tensor): [batch_size, chunk_size, embed_dim]
            target_chunk (torch.Tensor): [batch_size, chunk_size]

        Returns:
            torch.Tensor: Sum of cross-entropy loss for non-ignored tokens in the chunk
        """
        # Create mask for non-ignored tokens: [batch_size, chunk_size]
        mask = target_chunk != self.ignore_index

        # Select hidden states and targets where mask is True
        hidden_selected = hidden_chunk[mask]  # [num_valid, embed_dim]
        target_selected = target_chunk[mask]  # [num_valid]

        # Project only selected hidden states: [num_valid, embed_dim] @ [embed_dim, vocab_size]
        logits_selected = F.linear(hidden_selected, weight)  # [num_valid, vocab_size]

        # Compute cross-entropy on selected tokens
        return F.cross_entropy(
            logits_selected.float(), target_selected, reduction="sum"
        )

    def forward(
        self,
        weight: torch.Tensor,
        hidden: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            weight (torch.Tensor): Output layer weight, shape [vocab_size, embed_dim].
            hidden (torch.Tensor): Hidden states, shape [batch_size, seq_len, embed_dim].
            targets (torch.Tensor): Labels, shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Normalized cross-entropy loss.
        """
        # Total number of non-ignored tokens across the entire batch
        total_elements = (targets != self.ignore_index).sum()

        # Chunk hidden states and targets along sequence dimension
        hidden_chunks = hidden.chunk(self.num_output_chunks, dim=1)
        target_chunks = targets.chunk(self.num_output_chunks, dim=1)

        total_loss = 0.0
        for hidden_chunk, target_chunk in zip(hidden_chunks, target_chunks):
            total_loss += self.compute_cross_entropy(weight, hidden_chunk, target_chunk)

        return total_loss / total_elements


class ChunkedCrossEntropywithAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        weight: torch.Tensor,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        chunk_size: int = 512,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Forward pass for chunked cross-entropy loss.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context object to store information for backward pass
            weight (torch.Tensor): Shape [vocab_size, emb_dim]
            hidden (torch.Tensor): Shape [bsz, seq_len, emb_dim]
            targets (torch.Tensor): Shape [bsz, seq_len]
            chunk_size (int): Size of each chunk along sequence dimension
            ignore_index (int): Index to ignore in cross-entropy

        Returns:
            torch.Tensor: Normalized loss scalar
        """

        def compute_loss(hidden_chunk, weight, target_chunk, ignore_index):
            logits = hidden_chunk @ weight.t()
            logits = logits.float()
            return F.cross_entropy(
                logits, target_chunk, ignore_index=ignore_index, reduction="sum"
            )

        # Compute number of chunks based on batch size and sequence length
        bsz, seq_len, emb_dim = hidden.shape
        chunks = max(1, (bsz * seq_len) // chunk_size)

        # Split hidden and targets into chunks along sequence dimension
        hidden_chunks = hidden.chunk(chunks, dim=1)
        target_chunks = targets.chunk(chunks, dim=1)

        total_loss = 0.0
        total_valid_tokens = 0
        grad_inputs = []
        grad_weight = torch.zeros_like(weight) if weight.requires_grad else None

        # Process each chunk
        for hidden_chunk, target_chunk in zip(hidden_chunks, target_chunks):
            # Flatten for loss computation
            flat_hidden = hidden_chunk.reshape(
                -1, emb_dim
            )  # [bsz * chunk_seq_len, emb_dim]
            flat_target = target_chunk.reshape(-1)  # [bsz * chunk_seq_len]

            # Count valid tokens (non-ignored)
            valid_tokens = (flat_target != ignore_index).sum().item()
            total_valid_tokens += valid_tokens

            # Compute loss and gradients
            result = torch.func.grad_and_value(compute_loss)(
                flat_hidden, weight, flat_target, ignore_index
            )
            grads, chunk_loss = result
            total_loss += chunk_loss

            # Handle gradients
            if hidden.requires_grad:
                g_input = grads[0] if isinstance(grads, tuple) else grads
                # Reshape to original chunk shape: [bsz, chunk_seq_len, emb_dim]
                g_input = g_input.view(hidden_chunk.shape)
                grad_inputs.append(g_input)
            if weight.requires_grad:
                g_weight = (
                    grads[1] if isinstance(grads, tuple) and len(grads) > 1 else None
                )
                if g_weight is not None:
                    grad_weight.add_(g_weight)

        # Concatenate gradients along sequence dimension
        if grad_inputs:
            grad_input = torch.cat(grad_inputs, dim=1)  # [bsz, seq_len, emb_dim]
        else:
            grad_input = None

        # Normalize loss
        normalized_loss = (
            total_loss / total_valid_tokens if total_valid_tokens > 0 else total_loss
        )

        # Save for backward
        ctx.save_for_backward(grad_input, grad_weight)
        ctx.total_valid_tokens = total_valid_tokens

        return normalized_loss

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple:
        """
        Backward pass for chunked cross-entropy.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context object with saved tensors from forward pass
            grad_output (torch.Tensor): Gradient from upstream, typically scalar

        Returns:
            tuple: Tuple of gradients w.r.t. inputs (weight, hidden, targets, chunk_size, ignore_index)
        """
        grad_input, grad_weight = ctx.saved_tensors
        total_valid_tokens = ctx.total_valid_tokens

        # Scale gradients by grad_output / total_valid_tokens
        scale = grad_output / total_valid_tokens if total_valid_tokens > 0 else 0.0

        return (
            grad_weight * scale if grad_weight is not None else None,  # grad_weight
            grad_input * scale if grad_input is not None else None,  # grad_hidden
            None,  # grad_targets
            None,  # grad_chunk_size
            None,  # grad_ignore_index
        )


class ChunkedCrossEntropywithAutogradLoss(nn.Module):
    """Wrapper around ChunkedCEAutograd autograd function."""

    takes_hidden_input = True

    def __init__(self, chunk_size: int = 512, ignore_index: int = -100):
        super().__init__()
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index

    def forward(
        self, weight: torch.Tensor, hidden: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the loss module.

        Args:
            weight (torch.Tensor): Shape [vocab_size, emb_dim]
            hidden (torch.Tensor): Shape [bsz, seq_len, emb_dim]
            targets (torch.Tensor): Shape [bsz, seq_len]

        Returns:
            torch.Tensor: Normalized loss
        """
        return ChunkedCrossEntropywithAutograd.apply(
            weight, hidden, targets, self.chunk_size, self.ignore_index
        )
