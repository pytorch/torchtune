# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class GemmaNormEmbeddings(nn.Embedding):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(in_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return x * torch.tensor(self.out_dim**0.5, dtype=x.dtype)
