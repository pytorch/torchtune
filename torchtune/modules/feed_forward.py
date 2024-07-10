# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn, Tensor


class FeedForward(nn.Module):
    """This class implements the SwiGLU feedforward block as described in https://arxiv.org/pdf/2002.05202v1.

    Args:
        gate_proj (nn.Module): Projection from input dim to hidden dim, fed through activation
            and multiplied by up_proj.
        down_proj (nn.Module): Final projection to output dim.
        up_proj (nn.Module): Projection from input dim to hidden dim, multiplied by
            activation(gate_proj).
        activation (nn.Module): Activation function to use. Default is nn.SiLU().
    """

    def __init__(
        self,
        *,
        gate_proj: nn.Module,
        down_proj: nn.Module,
        up_proj: nn.Module,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.w1 = gate_proj
        self.w2 = down_proj
        self.w3 = up_proj
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.activation(self.w1(x)) * self.w3(x))


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        act_layer: nn.Module,
        dropout: int = 0.0,
    ):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.activation = act_layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.activation(x.float()).type_as(x)
        x = self.dropout(x)
        return self.layer2(x)
