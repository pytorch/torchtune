# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn

from .loss_protocols import SFTLinearLoss


class LinearCrossEntropyLoss(nn.Module, SFTLinearLoss):
    """Memory efficient Cross-entropy loss that incrementally computes loss for chunks of tokens
    by masking ignored tokens, calculating logits and then applying cross-entropy loss. Combines
    the linear projection with the cross-entropy calculation for futher memory savings.
    """

    def __init__(
        self,
        num_output_chunks: int = 8,
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
        self.ignore_index = ignore_index
        self.mask_pre_projection = mask_pre_projection

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the compute_cross_entropy function.
        If compiling CE + chunking operation together, memory requirement is higher."""
        self.compute_cross_entropy = torch.compile(
            self.compute_cross_entropy, *args, **kwargs
        )
        return self

    def compute_cross_entropy(
        self,
        weight: torch.Tensor,
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Computes cross-entropy by masking tokens, calculating logits and then applying cross-entropy loss.

        Args:
            weight (torch.Tensor): [vocab_size, embed_dim]
            hidden_chunk (torch.Tensor): [batch_size, chunk_size, embed_dim]
            target_chunk (torch.Tensor): [batch_size, chunk_size]
        Returns:
            torch.Tensor: Sum of cross-entropy loss for non-ignored tokens in the chunk
        """
        # Select hidden states and targets where mask is True
        if self.mask_pre_projection:
            mask_chunk = target_chunk != self.ignore_index
            hidden_chunk = hidden_chunk[mask_chunk]  # [num_valid, embed_dim]
            target_chunk = target_chunk[mask_chunk]  # [num_valid]
        else:
            hidden_chunk = hidden_chunk.reshape(-1, hidden_chunk.shape[-1])
            target_chunk = target_chunk.reshape(-1)

        # [num_valid, embed_dim] @ [embed_dim, vocab_size]
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
            weight (torch.Tensor): Tensor with weights of the model output projection layer. Shape ``[vocab_size, emb_dim]``
            outputs (torch.Tensor): Hidden state of the model, pre projection. Shape ``[bsz, seq_len, emb_dim]``
            targets (torch.Tensor): Labels for the model. Shape ``[bsz, seq_len]``
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            torch.Tensor: loss tensor
        """
        # Total number of non-ignored tokens across the entire batch
        mask = targets != self.ignore_index
        total_elements = mask.sum()

        # Chunk along sequence dimension
        hidden_chunks = outputs.tensor_split(self.num_output_chunks, dim=1)
        target_chunks = targets.tensor_split(self.num_output_chunks, dim=1)

        # Compute cross-entropy loss for the chunks
        total_loss = 0.0
        for idx in range(len(hidden_chunks)):
            total_loss += self.compute_cross_entropy(
                weight,
                hidden_chunks[idx],
                target_chunks[idx],
            )

        return total_loss / total_elements
