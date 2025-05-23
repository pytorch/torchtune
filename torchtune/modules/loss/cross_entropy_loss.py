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
        log.warning("Skipping compile loss, as it is not supported at this time")
        # TODO fix compile and re-enable
        # self.compute_cross_entropy = torch.compile(
        #     self.compute_cross_entropy, *args, **kwargs
        # )
        return self

    def set_model_output(self, model: nn.Module) -> None:
        """Modify model output to match the expected input for the loss function."""
        model.skip_output_layer = True
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
            AttributeError: if called before update_model
        """
        # Select hidden states and targets where mask is True
        mask_chunk = target_chunk != self.ignore_index
        if mask_chunk.sum() == 0:
            # Unmask 1 token to allow loss to sync with all data parallel workers
            mask_chunk[0] = True

        target_chunk = target_chunk[mask_chunk]  # [num_valid]
        if isinstance(hidden_chunk, DTensor):
            # DTensor doesn't support masks so we have to mask locally
            mesh = hidden_chunk.device_mesh
            placements = hidden_chunk.placements
            local_hidden_chunk = hidden_chunk.to_local()[mask_chunk]
            hidden_chunk = DTensor.from_local(
                local_hidden_chunk, mesh, placements
            )  # [num_valid, embed_dim]
        else:
            hidden_chunk = hidden_chunk[mask_chunk]  # [num_valid, embed_dim]

        # [num_valid, embed_dim] @ [embed_dim, vocab_size]
        if self.linear_projection is None:
            raise AttributeError("forward called before update_model")
        logits = self.linear_projection(hidden_chunk)  # [num_valid, vocab_size]
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()

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

        if total_elements == 0:
            # must return after calling compute_cross_entropy to not hang during data parallel training
            return total_loss
        else:
            return total_loss / total_elements


class LigerLinearCrossEntropy(nn.Module, SFTLoss):
    """Memory efficient Cross-entropy loss that uses fused CUDA kernels to compute the loss.
    Combines the linear projection with the cross-entropy calculation for better performance
    and memory efficiency. This is an approximation of CrossEntropyLoss and may have small
    numerical differences compared to the standard implementation.

    Args:
        ignore_index (int): Index to ignore in the target tensor. Default is -100.

    Raises:
        ImportError: If liger_kernel is not installed

    Note:
        This loss requires `liger_kernel` and `triton` to be installed:
        `pip install triton liger_kernel`

    Linear cross entropy is computed in a single fused operation. You need to skip the final
    projection layer in your model and pass it to the loss instead. You can setup the loss
    with the model as shown below.

    >>> model = Transformer(...)
    >>> loss = LigerLinearCrossEntropy(...)
    >>> loss.set_model_output(model)
    >>> loss.apply_compile_strategy()
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        try:
            from liger_kernel.ops.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyFunction,
            )

            self.fused_linear_ce = LigerFusedLinearCrossEntropyFunction
        except ImportError as err:
            raise ImportError(
                "liger_kernel is required for LigerLinearCrossEntropy but not installed. "
                "Please install it before using this loss function."
            ) from err
        self.linear_projection = None
        self.ignore_index = ignore_index

    def set_model_output(self, model: nn.Module) -> None:
        """Modify model output to match the expected input for the loss function.

        Args:
            model (nn.Module): The model whose output layer will be used for the loss computation.

        Raises:
            ValueError: If model.output doesn't have required weight and bias parameters
        """
        model.skip_output_layer = True
        self.linear_projection = model.output

        # Validate the projection layer has required parameters
        if not hasattr(self.linear_projection, "weight"):
            raise ValueError(
                "Model output layer must have a weight parameter for LigerLinearCrossEntropy"
            )

    def apply_compile_strategy(self, *args, **kwargs):
        """Triton kernels are already JIT-compiled so no additional compilation needed."""
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the fused linear cross entropy loss.

        Args:
            hidden_states (torch.Tensor): Hidden state of the model, pre projection. Shape ``[bsz, seq_len, emb_dim]``
            targets (torch.Tensor): Labels for the model. Shape ``[bsz, seq_len]``

        Returns:
            torch.Tensor: The computed loss value

        Raises:
            RuntimeError: If set_model_output() was not called before forward
        """
        if self.linear_projection is None:
            raise RuntimeError("Must call set_model_output() before forward()")
        if isinstance(hidden_states, DTensor):
            hidden_states = hidden_states.full_tensor()

        hidden_states = hidden_states.flatten(0, 1)  # [batch_size*seq_len, emb_dim]
        targets = targets.flatten()  # [batch_size*seq_len]

        w = self.linear_projection.weight
        if isinstance(w, DTensor):
            w = w.full_tensor()

        b = getattr(self.linear_projection, "bias", None)
        if b is not None and isinstance(b, DTensor):
            b = b.full_tensor()

        loss, _ = self.fused_linear_ce.apply(
            hidden_states,
            w,
            targets,
            b,
            None,
            self.ignore_index,
        )
        return loss
