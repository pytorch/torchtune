# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    """This class implements the SwiGlu feed-forward. For more details, see
    https://arxiv.org/pdf/2002.05202.

    Args:
        gate_proj (nn.Module): Projection from input dim to hidden dim, fed through activation
            and multiplied by up_proj.
        down_proj (nn.Module): Final projection to output dim.
        up_proj (Optional[nn.Module]): Projection from input dim to hidden dim, multiplied by
            activation(gate_proj).
        activation (nn.Module): Activation function to use. Default is nn.SiLU().
    """

    def __init__(
        self,
        *,
        gate_proj: nn.Module,
        down_proj: nn.Module,
        up_proj: Optional[nn.Module] = None,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.w1 = gate_proj
        self.w2 = down_proj
        self.w3 = up_proj
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.w1(x))
        if self.w3 is not None:
            h = h * self.w3(x)
        h = self.w2(h)
        return h


class TiedLinear:
    """
    A tied linear layer that shares the same weight as another linear layer.
    A module is required, instead of the weight of the module, so it
    can work with FSDP. Otherwise, the memory reference will be lost.
    """

    def __init__(self, tied_module: nn.Module):
        self.tied_module = tied_module

    def __call__(self, x: torch.tensor) -> torch.tensor:
        return F.linear(x, self.tied_module.weight)
