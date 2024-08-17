# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torch import nn, Tensor


class TanhGate(nn.Module):
    """Implements a basic learnable gate to scale layer outputs"""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor to gate

        Returns:
            Tensor: The output tensor after gating.
        """
        return x * self.scale.tanh()
