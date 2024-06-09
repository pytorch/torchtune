# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable
import torch

class LayerDropout(torch.nn.Module):
    def __init__(self, prob=0.0, dim=0, disable_on_eval=True):
        super().__init__()
        self.prob: float = prob
        self.dim = dim
        self.disable_on_eval: bool = disable_on_eval
        self.generator = torch.Generator(device="cpu")
        self.inferred: float = None

    def forward(self, function: Callable, input: torch.Tensor, *args, **kwargs):
        n = input.shape[self.dim]

        if self.prob == 0 or (self.disable_on_eval and self.training is False):
            self.inferred = 1.0
            return function(input, *args, **kwargs)

        skip = torch.bernoulli(torch.Tensor((n) * [self.prob]), generator=self.generator).to(input.device).to(input.dtype)
        self.inferred = 1 - torch.mean(skip)
        ind_selected = (skip == 0).nonzero().squeeze().to(input.device)

        if ind_selected.numel() > 0:
            x_selected = torch.index_select(input, self.dim, ind_selected)
            out_selected = function(x_selected, *args, **kwargs)

        out = input.clone()
        assert self.dim == 0, "Currently only supporting dropping elements along the 0th dimension"
        if ind_selected.numel() > 0:
            out[ind_selected] = out_selected
        return out
