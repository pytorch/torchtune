# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

from torch import nn, Tensor


class LoRALinear(nn.Module):
    """LoRA linear layer as introduced in https://arxiv.org/abs/2106.09685.

    LoRA perturbs a given layer via a low-rank approximation where only
    the rank decomposition matrices are trainable. In a linear layer instead of
    :math:`x \\mapsto W_0x` a LoRALinear layer is defined as
    :math:`x \\mapsto W_0x + (\\alpha / r)BAx`, where :math:`r` is the rank of
    the matrices :math:`A` and :math:`B` and :math:`\\alpha` is a scaling factor.
    As in the original implementation, we support dropout before multiplication
    by the low-rank matrices.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        rank (int): rank of the low-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability
        use_bias (bool): whether to include bias in the two low-rank matrices
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.out_dim = out_dim
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=use_bias)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=use_bias)
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/main/loralib/layers.py#L119
        nn.init.zeros_(self.lora_b.weight)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``
        """
        out = self.linear(x)
        lora_out = self.lora_a(self.dropout(x))
        lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)
        return out + lora_out
