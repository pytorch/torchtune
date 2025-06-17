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

    # makes subclasses with multiple inheritance including nn.Module play nicely
    # https://github.com/pytorch/pytorch/pull/91819
    call_super_init = True

    def __init__(self, *, enable_loss_parallel: bool = False):
        super().__init__()
        self.enable_loss_parallel = enable_loss_parallel

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
    @abstractmethod
    def supports_loss_parallel(self) -> bool:
        """
        Whether the loss function supports loss parallel.
        Set to false if loss parallelism isn't tested with your loss class.
        """
        pass

    @property
    @abstractmethod
    def loss_parallel_requires_ctx_manager(self) -> bool:
        """
        Whether to use the loss parallel context manager for loss parallelism. Can be
        used if the function relies on the standard cross_entropy() or CrossEntropyLoss.
        Set to false if loss parallelism isn't tested with your loss class, or your loss
        parallelism doesn't require the context manager.
        """
        pass

    def patch_tp_plan(self, tp_plan) -> bool:
        """Whether the loss function supports loss parallel. Defaults to a noop."""
        return tp_plan

    @property
    def loss_parallel_enabled(self) -> bool:
        """
        The `enable_loss_parallel` flag is a directive from the user.
        This property also validates that it is supported, or raises an error.
        """
        if self.enable_loss_parallel and not self.supports_loss_parallel:
            raise ValueError(
                f"Loss function is enabled, but {self.__class__.__name__} does not support loss parallelism"
            )
        return self.enable_loss_parallel

    @property
    def use_loss_parallel_ctx_manager(self) -> bool:
        """
        Whether to enable the loss parallelism ctx manager for this instance of the class.
        """
        return self.loss_parallel_enabled and self.loss_parallel_requires_ctx_manager


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
