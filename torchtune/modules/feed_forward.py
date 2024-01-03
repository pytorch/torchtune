# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
from torch import nn, Tensor


class FeedForward(nn.Module):
    """This class implements the feed-forward network derived from Llama2.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        linear_class (nn.Module): Linear module, can we switch with a custom implementation, e.g. LoRALinear.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        linear_class: nn.Module,
    ):
        super().__init__()
        self.w1 = linear_class(dim, hidden_dim, bias=False)
        self.w2 = linear_class(hidden_dim, dim, bias=False)
        self.w3 = linear_class(dim, hidden_dim, bias=False)
        self.activation = F.silu

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.activation(self.w1(x)) * self.w3(x))
