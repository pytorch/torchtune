# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import nn


class SFTLoss(ABC):
    """Interface for loss functions in torchtune used in sft recipes."""

    def apply_compile_strategy(self, *args, **kwargs):
        """Compile the loss function for inference."""
        self.forward = torch.compile(self.forward, *args, **kwargs)
        return self

    @abstractmethod
    def set_model_output(self, model: nn.Module) -> None:
        """Modify model output to match the expected input for the loss function."""
        pass

    @abstractmethod
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            outputs (torch.Tensor): Logits of the model. Shape [bsz, seq_len, vocab_size]
            targets (torch.Tensor): Labels for the model. Shape [bsz, seq_len]

        Returns:
            torch.Tensor: loss tensor
        """
        pass

    @property
    def tp_requires_loss_parallel_ctx_manager(self) -> bool:
        """
        Whether to use the loss parallel context manager for loss parallelism.
        https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.loss_parallel
        """
        return False

    def patch_tp_plan(self, tp_plan) -> dict:
        """Whether the loss function supports loss parallel. Defaults to a noop."""
        return tp_plan


class RLLoss(ABC):
    """Interface for loss functions in torchtune used in RL recipes."""

    def apply_compile_strategy(self, *args, **kwargs):
        """Torch compiles the loss function. Can be useful when greater control is needed,
        for example when only compiling a portion of the loss calculation."""
        self.forward = torch.compile(self.forward, *args, **kwargs)
        return self

    @abstractmethod
    def set_model_output(self, model: nn.Module) -> None:
        """Modify model output to match the expected input for the loss function."""
        pass

    @abstractmethod
    def forward(
        self,
        pi_old_outputs: torch.Tensor,  # [B x G, L]
        pi_outputs: torch.Tensor,  # [B x G, L]
        ref_outputs: torch.Tensor,  # [B x G, L]
        advantages: torch.Tensor,  # [B x G]
        padding_masks: Optional[torch.Tensor] = None,  # [B x G, L]
    ) -> Any:
        """
        Args:
            pi_old_outputs (torch.Tensor): Outputs of the old policy. Shape ``[B x G, L]``
            pi_outputs (torch.Tensor): Outputs of the new policy. Shape ``[B x G, L]
            ref_outputs (torch.Tensor): Outputs of the reference policy. Shape ``[B x G, L
            advantages (torch.Tensor): Advantages of the new policy. Shape ``[B x G]``
            padding_masks (Optional[torch.Tensor]): Mask for padding tokens. Shape ``[B x G, L]``

        Returns:
            Any: Object containing the relevant loss information
        """
        pass
