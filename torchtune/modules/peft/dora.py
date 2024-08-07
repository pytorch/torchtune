# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List

import torch
import torch.nn.functional as F

from torch import nn, Tensor

from torchao.dtypes.nf4tensor import linear_nf4, to_nf4
from torchtune.modules.low_precision import _register_nf4_dispatch_ops  # noqa: F401
from torchtune.modules.peft.peft_utils import AdapterModule


class DoRALinear(nn.Module, AdapterModule):
    """LoRA linear layer as introduced in `LoRA: Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2106.09685>`_.

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
        dropout (float): dropout probability. Default: 0.0
        use_bias (bool): whether to include bias in the original linear layer.
            Default: False
        quantize_base (bool): Whether to quantize base linear weight or not.
            Default: False

    Raises:
        NotImplementedError: If use_bias is enabled.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
        quantize_base: bool = False,
    ):
        super().__init__()
        if use_bias:
            raise NotImplementedError("DoRALinear does not support using bias")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scaling = alpha / rank
        self._quantize_base = quantize_base
        weight = self._create_weight()
        self.register_parameter("weight", nn.Parameter(weight))

        # 'self.disabled' is a flag showing whether to turn off DoRA adapters,
        # this can be used in DPO for treating the dora adapters as the policy model
        # and disabling it to treat the base model as the reference model
        self.disabled = False

        self.dropout = nn.Dropout(p=dropout)
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        self.magnitude = nn.Parameter(torch.empty(out_dim))
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def initialize_dora_magnitude(self):
        """
        DoRA initializes the magnitude vector such that its outputs are initially
        identical to standard LoRA's outputs.
        """
        base_weight = self.weight.to(self.lora_a.weight.dtype)
        lora_weight = self.lora_b.weight @ self.lora_a.weight
        weight = base_weight + self.scaling * lora_weight
        magnitude = torch.linalg.norm(weight, dim=1)
        self.magnitude = nn.Parameter(magnitude)

    def _create_weight(self):
        """
        Creates a linear weight and bias tensor, using NF4 dtype if we're quantizing
        (indicated via quantize_base=True).
        """
        in_dim, out_dim = self.in_dim, self.out_dim
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=False)
        weight = linear.weight if not self._quantize_base else to_nf4(linear.weight)
        return weight

    def adapter_params(self) -> List[str]:
        """
        Return lora_a.weight and lora_b.weight as adapter params.
        If bias is enabled, also return lora_a.bias and lora_b.bias.
        """
        # NOTE: this function has to be updated if the names of "lora_a" and "lora_b"
        # in this module change.
        # TODO: need to add back magnitude, but causing initial check to error out cause it's not defined yet
        adapter_params = ["lora_a.weight", "lora_b.weight", "magnitude"]
        return adapter_params

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``
        """
        if self._quantize_base:
            base_out = linear_nf4(input=x, weight=self.weight)
        else:
            base_out = F.linear(x, self.weight)
        if self.disabled:
            return base_out

        x = self.dropout(x)

        # Can't use raw matmul since FSDP hooks are attached to __call__
        # Instead follow the approach in https://github.com/huggingface/peft/pull/1806
        x_eye = torch.eye(
            self.lora_a.weight.shape[1], device=self.lora_a.weight.device, dtype=x.dtype
        )
        lora_weight = self.lora_b(self.lora_a(x_eye)).T

        weight = self.weight.to(x.dtype) + self.scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).detach()
        mag_norm_scale = (self.magnitude / weight_norm).view(1, -1)
        lora_out = self.lora_a(x)
        lora_out = self.scaling * self.lora_b(lora_out)
        dora_out = (mag_norm_scale - 1) * base_out + mag_norm_scale * lora_out
        return dora_out

        return self.lora(x, base_out, self.weight)
