# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from .loss_types import SFTLoss


class LinearCrossEntropyLoss(nn.Module, SFTLoss):
    """Memory efficient Cross-entropy loss that incrementally computes loss for chunks of tokens
    by masking ignored tokens, calculating logits and then applying cross-entropy loss. Combines
    the linear projection with the cross-entropy calculation for futher memory savings.

    Linear cross entropy entropy masks out ignored tokens before the projection layer to save memory.
    You therefore need to skip the final projection layer in your model and pass it to the loss instead.
    You can setup the loss with the model and compile it as shown below.

    >>> model = Transformer(...)
    >>> loss = LinearCrossEntropyLoss(...)
    >>> loss.set_model_output(model)
    >>> loss.apply_compile_strategy()
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
        self.linear_projection = None
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

    def set_model_output(self, model: nn.Module) -> None:
        """Modify model output to match the expected input for the loss function."""
        model.skip_output = True
        self.linear_projection = model.output

    def compute_cross_entropy(
        self,
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Computes cross-entropy by masking tokens, calculating logits and then applying cross-entropy loss.

        Args:
            hidden_chunk (torch.Tensor): [batch_size, chunk_size, embed_dim]
            target_chunk (torch.Tensor): [batch_size, chunk_size]

        Returns:
            torch.Tensor: Sum of cross-entropy loss for non-ignored tokens in the chunk

        Raises:
            TypeError: if you use this method with a DTensor
            AttributeError: if called before update_model
        """
        # Select hidden states and targets where mask is True
        if self.mask_pre_projection:
            if isinstance(hidden_chunk, DTensor):
                raise TypeError(
                    "LinearCrossEntropyLoss doesn't work with distributed models. Please use CEWithChunkedOutputLoss."
                )
                # TODO: Fix linear_loss for TP models
                # target_chunk = distribute_tensor(
                #     target_chunk,
                #     hidden_chunk.device_mesh,
                # )
            mask_chunk = target_chunk != self.ignore_index
            hidden_chunk = hidden_chunk[mask_chunk]  # [num_valid, embed_dim]
            target_chunk = target_chunk[mask_chunk]  # [num_valid]
        else:
            hidden_chunk = hidden_chunk.reshape(-1, hidden_chunk.shape[-1])
            target_chunk = target_chunk.reshape(-1)

        # [num_valid, embed_dim] @ [embed_dim, vocab_size]
        if self.linear_projection is None:
            raise AttributeError("forward called before update_model")
        logits = self.linear_projection(hidden_chunk)  # [num_valid, vocab_size]
        # TODO: fix linear_loss for TP models
        # if isinstance(logits, DTensor):
        #    logits = logits.full_tensor()

        return F.cross_entropy(
            logits.float(),
            target_chunk,
            reduction="sum",
            ignore_index=self.ignore_index,
        )

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            outputs (torch.Tensor): Hidden state of the model, pre projection. Shape ``[bsz, seq_len, emb_dim]``
            targets (torch.Tensor): Labels for the model. Shape ``[bsz, seq_len]``

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
                hidden_chunks[idx],
                target_chunks[idx],
            )

        return total_loss / total_elements
