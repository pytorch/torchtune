# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
from torch import nn, Tensor


class FeedForward(nn.Module):
    """This class implements the feed-forward network of LLaMA.
    Notably, this utilizes a variant of GLU called SwiGLU, a combination of Swish
    and Gated Linear Units, formulated in https://arxiv.org/pdf/2002.05202.pdf (Shazeer 2020).

    Reference implementation (used for correctness verfication) can be
    found here: https://github.com/facebookresearch/llama/model.py#L307.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer. `hidden_dim` should be a large multiple of 2
            and can additionally include a custom multiplier, which affects the model capacity for LLaMA V2. In
            the original facebookresearch/llama documentation, this is referred to as `ffn_custom_multiplier`.
        multiple_of (int): After applying a SwiGLU scaling factor to `hidden_dim`, `hidden_dim` is rounded to
            the nearest multiple of `multiple_of` that is greater than `hidden_dim`. Based on experiments, the
            LLaMA team recommends rounding the result to a multiple of 256. E.g. for LLaMA 7B, the hidden dimension
            would be 11008 after applying SwiGLU-related scaling and rounding. Default: 256
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
    ):
        super().__init__()
        # Scale hidden dimension by (2/3)4d for SwiGLU to keep number of
        # parameters and computation constant
        hidden_dim = 4 * int(2 * hidden_dim / 3)

        # Round hidden dimension to nearest multiple of `multiple_of`
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU = W_2(Swish(Wx) ⊗ Vx) from Shazeer 2020
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
