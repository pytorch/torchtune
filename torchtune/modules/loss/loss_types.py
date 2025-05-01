# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

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
