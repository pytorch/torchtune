# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import ColwiseParallel

from torchtune.modules.loss.loss_types import SFTLoss
from torchtune.utils import get_logger

log = get_logger()


class LinearCrossEntropyLoss(SFTLoss, nn.Module):
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
        enable_loss_parallel: bool = False,
        mask_ignored_tokens: bool = True,
    ):
        super().__init__(enable_loss_parallel=enable_loss_parallel)
        """
        Args:
            num_output_chunks (int): Number of chunks to split the output tensor into. Default is 8.
            ignore_index (int): Index to ignore in the target tensor. Default is -100.
            enable_loss_parallel (bool): Whether to enable loss parallel. Default is False.
            mask_ignored_tokens (bool): Whether to mask out ignored tokens during loss computation. Default is True.
        """
        self.linear_projection = None
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.mask_ignored_tokens = mask_ignored_tokens

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the compute_cross_entropy function.
        If compiling CE + chunking operation together, memory requirement is higher."""
        # we might be able to compile in TP case if masking is disabled?
        if not self.loss_parallel_enabled or not self.mask_ignored:
            self.compute_cross_entropy = torch.compile(
                self.compute_cross_entropy, *args, **kwargs
            )
        else:
            log.warning(
                "Skipping compile loss, as it is not supported with loss parallel enabled."
            )
        return self

    def set_model_output(self, model: nn.Module) -> None:
        """Modify model output to match the expected input for the loss function."""
        model.skip_output_layer = True
        self.linear_projection = model.output

    def patch_tp_plan(self, tp_plan) -> bool:
        if self.loss_parallel_enabled:
            if "output" not in tp_plan:
                raise KeyError("`tp_plan` requires `output` key")

            tp_plan["output"] = ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1),
                use_local_output=False,
            )
        return tp_plan

    @property
    def supports_loss_parallel(self) -> bool:
        return True

    @property
    def loss_parallel_requires_ctx_manager(self) -> bool:
        return True

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
        # [num_valid, embed_dim] @ [embed_dim, vocab_size]
        if self.linear_projection is None:
            raise AttributeError("forward called before update_model")
        logits = self.linear_projection(hidden_chunk)  # [num_valid, vocab_size]

        loss = F.cross_entropy(
            logits.float(),
            target_chunk,
            reduction="sum",
            ignore_index=self.ignore_index,
        )
        return loss

    def mask_inputs(
        self, hidden: torch.Tensor, target: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden (torch.Tensor): Hidden state of the model, pre projection. Shape ``[bsz, seq_len, emb_dim]``
            target (torch.Tensor): Labels for the model. Shape ``[bsz, seq_len]``
            indices (torch.Tensor): Indices of the valid tokens. Shape ``[num_valid]``

        Returns:
            tuple[torch.Tensor, torch.Tensor]: returns a tuple of
            - The indexed hidden states
            - The indexed targets
        """
        if isinstance(hidden, DTensor):
            device_mesh = hidden.device_mesh
            hidden = hidden.to_local().index_select(0, indices)
            hidden = DTensor.from_local(
                hidden,
                device_mesh=device_mesh,
                placements=[Shard(-1)] * device_mesh.ndim,
            )
        else:
            hidden = hidden.index_select(0, indices)

        target = target.index_select(0, indices)
        return hidden, target

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
        total_valid_tokens = torch.where(targets != self.ignore_index)[0].numel()
        if total_valid_tokens == 0:
            return torch.tensor(0.0, device=targets.device)

        # this redistribute allows tensor spitting without replication
        if isinstance(outputs, DTensor):
            outputs = outputs.redistribute(
                device_mesh=outputs.device_mesh,
                placements=[Shard(-1)] * outputs.device_mesh.ndim,
            )

        targets = targets.reshape(-1)
        outputs = outputs.reshape(-1, outputs.shape[-1])

        if self.mask_ignored:
            indices = torch.where(targets != self.ignore_index)[0]
            outputs, targets = self.mask_inputs(outputs, targets, indices)

        hidden_chunks = outputs.tensor_split(self.num_output_chunks, dim=0)
        target_chunks = targets.tensor_split(self.num_output_chunks, dim=0)

        total_loss = torch.tensor(0.0, device=targets.device)
        for hidden_chunk, target_chunk in zip(hidden_chunks, target_chunks):
            loss = self.compute_cross_entropy(hidden_chunk, target_chunk)
            # without this backprop throws `'Tensor' object has no attribute '_local_tensor'`
            if isinstance(loss, DTensor):
                loss = loss.full_tensor()
            total_loss += loss

        return total_loss / total_valid_tokens
