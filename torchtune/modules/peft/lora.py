# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torch import nn, Tensor


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        rank: int,
        out_dim: int,
        alpha: float,
        use_bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=use_bias)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=use_bias)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        out = self.dropout(out)
        lora_out = self.lora_a(x)
        lora_out = (alpha / r) * self.lora_b(lora_out)
        return out + lora_out
