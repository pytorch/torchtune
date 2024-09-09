# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class EmbeddingNorm(nn.Module):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        hidden_dim = h.size(-1)
        h = h * torch.tensor(hidden_dim**0.5, dtype=h.dtype)
        return h
