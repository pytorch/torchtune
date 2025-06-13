# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtune.modules.loss.loss_types import SFTLoss
from torchtune.utils import get_logger

log = get_logger()


class LinearCrossEntropyLoss(nn.Module, SFTLoss):
    """Memory efficient Cross-entropy loss that incrementally computes loss for chunks of tokens
    by masking ignored tokens, calculating logits and then applying cross-entropy loss. Combines
    the linear projection with the cross-entropy calculation for further memory savings.

    This implementation is compatible with ``torch.compile`` and ``DTensor`` by separating
    the core computation from the distributed logic. It operates on local tensors and
    aggregates the results at the end.

    Linear cross entropy masks out ignored tokens before the projection layer to save memory.
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
    ):
        super().__init__()
        """
        Args:
            num_output_chunks (int): Number of chunks to split the output tensor into. Default is 8.
            ignore_index (int): Index to ignore in the target tensor. Default is -100.
        """
        self.linear_projection = None
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the compute_cross_entropy function.
        If compiling CE + chunking operation together, memory requirement is higher."""
        self.compute_cross_entropy = torch.compile(
            self.compute_cross_entropy, *args, **kwargs
        )
        return self

    def set_model_output(self, model: nn.Module) -> None:
        """Modify model output to match the expected input for the loss function."""
        model.skip_output_layer = True
        self.linear_projection = model.output

    def compute_cross_entropy(
        self,
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes cross-entropy by masking tokens, calculating logits and then applying cross-entropy loss.

        Args:
            hidden_chunk (torch.Tensor): [batch_size, chunk_size, embed_dim]
            target_chunk (torch.Tensor): [batch_size, chunk_size]
        Returns:
            tuple[torch.Tensor, torch.Tensor]: returns a tuple of
            - The sum of the cross-entropy loss for the valid tokens in the chunk.
            - The number of valid tokens in the chunk.
        Raises:
            AttributeError: If the linear projection is not set, an AttributeError is raised.
        """
        hidden_flat = hidden_chunk.reshape(-1, hidden_chunk.size(-1))
        target_flat = target_chunk.reshape(-1)
        mask_flat = target_flat != self.ignore_index

        # Get indices of valid (non-ignored) tokens
        valid_indices = torch.where(mask_flat)[0]

        if valid_indices.numel() == 0:
            return (
                torch.tensor(0.0, device=hidden_chunk.device),
                torch.tensor(0, device=hidden_chunk.device),
            )

        # Mask: Select hidden states and targets for valid tokens using index_select.
        # (where+index_select works better with compile than boolean indexing)
        valid_hidden = torch.index_select(hidden_flat, 0, valid_indices)
        valid_targets = torch.index_select(target_flat, 0, valid_indices)

        # [num_valid, embed_dim] @ [embed_dim, vocab_size]
        if self.linear_projection is None:
            raise AttributeError(
                "Loss function was not properly configured. "
                "Ensure set_model_output() is called before the forward pass."
            )
        logits = self.linear_projection(valid_hidden)  # [num_valid, vocab_size]

        loss_sum = F.cross_entropy(
            logits.float(),
            valid_targets,
            reduction="sum",
            ignore_index=self.ignore_index,
        )
        num_valid_tokens = torch.tensor(
            valid_indices.numel(), device=hidden_chunk.device
        )
        return loss_sum, num_valid_tokens

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            outputs (torch.Tensor): Hidden state of the model, pre projection.
                Can be a ``DTensor``. Shape ``[bsz, seq_len, emb_dim]``
            targets (torch.Tensor): Labels for the model. Shape ``[bsz, seq_len]``

        Returns:
            torch.Tensor: The final, averaged loss tensor.
        """
        # --- DTensor Handling ---
        # Check if the input is a DTensor and convert to local tensor.
        # All subsequent operations are performed on local tensors, making them
        # torch.compile / masking friendly.
        is_dtensor_input = isinstance(outputs, DTensor)
        if is_dtensor_input:
            local_outputs = outputs.to_local()
        else:
            local_outputs = outputs

        # --- Chunk along sequence dimension ---
        hidden_chunks = local_outputs.tensor_split(self.num_output_chunks, dim=1)
        target_chunks = targets.tensor_split(self.num_output_chunks, dim=1)

        # Compute cross-entropy loss for the chunks
        total_loss = torch.tensor(0.0, device=local_outputs.device)
        total_valid_tokens = torch.tensor(0, device=local_outputs.device)
        for hidden_chunk, target_chunk in zip(hidden_chunks, target_chunks):
            loss_sum, num_valid_tokens = self.compute_cross_entropy(
                hidden_chunk, target_chunk
            )
            total_loss += loss_sum
            total_valid_tokens += num_valid_tokens

        # --- Mean reduce ---
        # If total_valid_tokens is 0, return a zero loss to avoid division by zero.
        # Use torch.where to avoid calling .item().
        return torch.where(
            total_valid_tokens > 0,
            total_loss / total_valid_tokens,
            total_loss,  # This will be 0.0 if total_valid_tokens is 0
        )
