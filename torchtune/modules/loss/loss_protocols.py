# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Protocol

import torch


class SFTLinearLoss(Protocol):
    """Protocol for loss functions in torchtune used in Supervised Finetune recipes that require
    model output linear projection weights in loss computation."""

    linear_loss: bool = True

    def apply_compile_strategy(self, *args, **kwargs):
        """Torch compiles the loss function. Can be useful when greater control is needed,
        for example when only compiling a portion of the loss calculation."""
        self.forward = torch.compile(self.forward, *args, **kwargs)
        return self

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
        ...


class RLLinearLoss(Protocol):
    """Protocol for loss functions in torchtune used in RL recipes that require
    model output linear projection weights in loss computation."""

    linear_loss: bool = True

    def apply_compile_strategy(self, *args, **kwargs):
        """Torch compiles the loss function. Can be useful when greater control is needed,
        for example when only compiling a portion of the loss calculation."""
        self.forward = torch.compile(self.forward, *args, **kwargs)
        return self

    def forward(
        self,
        pi_old_logprobs: torch.Tensor,  # [B x G, L]
        pi_logprobs: torch.Tensor,  # [B x G, L]
        ref_logprobs: torch.Tensor,  # [B x G, L]
        advantages: torch.Tensor,  # [B x G]
        padding_masks: Optional[torch.Tensor] = None,  # [B x G, L]
        *args,
        **kwargs,
    ) -> Any:
        """
        Args:
            pi_old_logprobs (torch.Tensor): Log probabilities of the old policy. Shape ``[B x G, L]``
            pi_logprobs (torch.Tensor): Log probabilities of the new policy. Shape ``[B x G, L]
            ref_logprobs (torch.Tensor): Log probabilities of the reference policy. Shape ``[B x G, L
            advantages (torch.Tensor): Advantages of the new policy. Shape ``[B x G]``
            padding_masks (Optional[torch.Tensor]): Mask for padding tokens. Shape ``[B x G, L]``
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Object containing the relevant loss information
        """
        ...


class SFTLoss(Protocol):
    """Protocol for loss functions in torchtune used in sft recipes."""

    linear_loss: bool = False

    def apply_compile_strategy(self, *args, **kwargs):
        """Compile the loss function for inference."""
        self.forward = torch.compile(self.forward, *args, **kwargs)
        return self

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            outputs (torch.Tensor): Logits of the model. Shape [bsz, seq_len, vocab_size]
            targets (torch.Tensor): Labels for the model. Shape [bsz, seq_len]
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            torch.Tensor: loss tensor
        """
        ...
