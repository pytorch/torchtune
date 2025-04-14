# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn

from .loss_protocols import SFTLossWithProjection


class ChunkedCrossEntropyLoss(nn.Module, SFTLossWithProjection):
    """Cross-entropy loss that incrementally computes loss for chunks of tokens
    by masking, calculating logits and then applying cross-entropy loss"""

    def __init__(self, num_output_chunks: int = 8, ignore_index: int = -100):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index

    def compute_cross_entropy(
        self,
        weight: torch.Tensor,
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
        mask_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Computes cross-entropy by masking tokens, calculating logits and then applying cross-entropy loss.

        Args:
            weight (torch.Tensor): [vocab_size, embed_dim]
            hidden_chunk (torch.Tensor): [batch_size, chunk_size, embed_dim]
            target_chunk (torch.Tensor): [batch_size, chunk_size]
            mask_chunk (torch.Tensor): [batch_size, chunk_size]
        Returns:
            torch.Tensor: Sum of cross-entropy loss for non-ignored tokens in the chunk
        """
        # Select hidden states and targets where mask is True
        hidden_chunk = hidden_chunk[mask_chunk]  # [num_valid, embed_dim]
        target_chunk = target_chunk[mask_chunk]  # [num_valid]

        # Project only selected hidden states: [num_valid, embed_dim] @ [embed_dim, vocab_size]
        logits = F.linear(hidden_chunk, weight)  # [num_valid, vocab_size]

        return F.cross_entropy(
            logits.float(),
            target_chunk,
            reduction="sum",
            ignore_index=self.ignore_index,
        )

    def forward(
        self,
        weight: torch.Tensor,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            weight (torch.Tensor): Tensor with weights of the model output projection layer. Shape [vocab_size, emb_dim]
            outputs (torch.Tensor): Hidden state of the model, pre projection. Shape [bsz, seq_len, emb_dim]
            targets (torch.Tensor): Labels for the model. Shape [bsz, seq_len]
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            torch.Tensor: loss tensor
        """
        # Total number of non-ignored tokens across the entire batch
        mask = targets != self.ignore_index
        total_elements = mask.sum()

        # Chunk along sequence dimension
        hidden_chunks = outputs.chunk(self.num_output_chunks, dim=1)
        target_chunks = targets.chunk(self.num_output_chunks, dim=1)
        mask_chunks = mask.chunk(self.num_output_chunks, dim=1)

        total_loss = 0.0
        for idx in range(len(hidden_chunks)):
            # Compute cross-entropy loss for the chunk
            total_loss += self.compute_cross_entropy(
                weight, hidden_chunks[idx], target_chunks[idx], mask_chunks[idx]
            )

        return total_loss / total_elements
