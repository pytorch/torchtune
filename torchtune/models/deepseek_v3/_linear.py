# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Optional, Optional
from torchtune.modules import RMSNorm

class DeepSeekV3LatentLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        rope_head_dim: Optional[int] = None,
    ):
        super().__init__()
        self.rope_head_dim = rope_head_dim
        intermediate_dim = rope_head_dim + rank if rope_head_dim else rank
        self.a_proj = nn.Linear(
            in_features=in_dim, out_features=intermediate_dim, bias=False
        )
        self.b_proj = nn.Linear(in_features=intermediate_dim, out_features=out_dim, bias=False)
        self.norm = RMSNorm(rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s_x, _ = x.shape
        out = self.a_proj(x)

        if self.rope_head_dim:
            out, rope_out = torch.split(out, [self.rank, self.rope_head_dim], dim=-1)
            rope_out = rope_out.view(b, s_x, 1, self.rope_head_dim).transpose(1, 2)
            out = self.b_proj(self.norm(out))
            return out, rope_out

        return out
