# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor


def apply_swiglu(x: Tensor, w: nn.Module, v: nn.Module, w2: nn.Module) -> Tensor:
    """Applies SwiGLU as formulated in: https://arxiv.org/pdf/2002.05202.pdf.

    Args:
        x (Tensor): Input tensor.
        w (nn.Module): Linear transform used for Swish gated function.
        v (nn.Module): Vanilla linear transform.
        w2 (nn.Module): Overall linear transform.

    Returns:
        Transformed tensor.
    """
    return w2(F.silu(w(x)) * v(x))


class FeedForward(nn.Module):
    """This class implements the feed-forward network of LLaMA.
    Notably, this utilizes a variant of GLU called SwiGLU, a combination of Swish
    and Gated Linear Units.

    Reference implementation (used for correctness verfication) can be
    found here: https://github.com/facebookresearch/llama/model.py#L307.

     Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        hidden_dim_multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. This increases
            model capacity for LLaMA V2. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        hidden_dim_multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # round `hidden_dim` to nearest greater `hidden_dim_multiple_of`
        hidden_dim = hidden_dim_multiple_of * (
            (hidden_dim + hidden_dim_multiple_of - 1) // hidden_dim_multiple_of
        )

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return apply_swiglu(x, self.w1, self.w3, self.w2)
