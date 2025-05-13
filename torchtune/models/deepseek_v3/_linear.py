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
        *,
        in_dim: int,
        out_dim: int,
        rank: int,
        norm: nn.Module,
        rope_head_dim: Optional[int] = None,
    ):
        super().__init__()
        self.rope_head_dim = rope_head_dim or 0 
        self.a = nn.Linear(
            in_features=in_dim, out_features=rank + self.rope_head_dim, bias=False
        )
        self.b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s_x, _ = x.shape
        out = self.a(x)

        if self.rope_head_dim:
            out, rope_out = torch.split(out, [self.rank, self.rope_head_dim], dim=-1)
            rope_out = rope_out.view(b, s_x, 1, self.rope_head_dim).transpose(1, 2)
            out = self.b(self.norm(out))
            return out, rope_out

        return self.b(self.norm(out))
