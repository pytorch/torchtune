# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List

import torch
import torch.nn.functional as F

from torch import nn

from torchao.dtypes.nf4tensor import linear_nf4, to_nf4
from torchtune.modules.low_precision import _register_nf4_dispatch_ops  # noqa: F401
from torchtune.modules.peft import AdapterModule


class DoRALinear(nn.Module, AdapterModule):
    """DoRA linear layer as introduced in
    `DoRA: Weight-Decomposed Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2402.09353>`_.

    DoRA (Weight-Decomposed Low-Rank Adaptation) fine-tunes a layer by decomposing the pre-trained weights
    into two components: magnitude and direction. The magnitude component is a learnable scalar vector
    that scales each output channel, while the direction component, modified via LoRA, adjusts the orientation
    of weights. By scaling the LoRA update component :math:`BAx` with the `magnitude` vector, DoRA allows the model
    to apply distinct scaling adjustments across different output dimensions.

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
        **quantization_kwargs: Keyword arguments to pass to `to_nf4` when quantizing the base linear weight.
            Examples of valid arguments are `block_size` and `scaler_block_size`, which control the granularity of
            weight quantization and scaler quantization respectively. This is only used if `quantize_base` is True.
            Default None

    Raises:
        ValueError: If ``quantize_base`` is False, but quantization kwargs are provided.

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
        **quantization_kwargs,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scaling = alpha / rank
        self.use_bias = use_bias
        self._quantize_base = quantize_base

        if not self._quantize_base and any([v for v in quantization_kwargs.values()]):
            raise ValueError(
                f"``quantize_base`` is False, but received the following quantization arguments: {quantization_kwargs}"
            )

        # Setup weight and bias
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=self.use_bias)
        weight = (
            linear.weight
            if not self._quantize_base
            else to_nf4(linear.weight, **quantization_kwargs)
        )
        bias = linear.bias if self.use_bias else None

        # 'self.disabled' is a flag showing whether to turn off DoRA adapters,
        # this can be used in DPO for treating the dora adapters as the policy model
        # and disabling it to treat the base model as the reference model
        self.disabled = False
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias) if bias is not None else None
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        self.magnitude = nn.Parameter(torch.empty(out_dim))
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        _lora_a_init_params(self.lora_a)
        _lora_b_init_params(self.lora_b)

    @torch.no_grad()
    def initialize_dora_magnitude(self):
        """
        DoRA initializes the magnitude vector such that its outputs are initially
        identical to standard LoRA's outputs.
        """
        base_weight = self.weight.to(self.lora_a.weight.dtype)
        lora_weight = self.lora_b.weight @ self.lora_a.weight
        weight_norm = self._get_weight_norm(base_weight, lora_weight)
        self.magnitude.copy_(weight_norm)

    def _get_weight_norm(self, weight, lora_weight):
        weight = weight + self.scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def adapter_params(self) -> List[str]:
        """
        Return lora_a.weight and lora_b.weight as adapter params.
        If bias is enabled, also return lora_a.bias and lora_b.bias.
        """
        adapter_params = ["lora_a.weight", "lora_b.weight", "magnitude"]
        return adapter_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``
        """
        if self._quantize_base:
            base_out = linear_nf4(input=x, weight=self.weight)
            if self.use_bias:
                base_out = base_out + self.bias
        else:
            base_out = F.linear(x, self.weight, self.bias)
        if self.disabled:
            return base_out

        x = self.dropout(x)

        lora_out = self.lora_b(self.lora_a(x))
        # Can't use raw matmul since FSDP hooks are attached to __call__
        # Instead follow the approach in https://github.com/huggingface/peft/pull/1806
        x_eye = torch.eye(
            self.lora_a.weight.shape[1], device=self.lora_a.weight.device, dtype=x.dtype
        )
        lora_weight = self.lora_b(self.lora_a(x_eye)).T
        magnitude = self.magnitude
        weight = self.weight.to(x.dtype)
        weight_norm = self._get_weight_norm(weight, lora_weight.detach())
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)

        dora_out = (
            mag_norm_scale - 1
        ) * base_out + mag_norm_scale * lora_out * self.scaling

        return dora_out + base_out


def _lora_a_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA A weight to Kaiming uniform.
    """
    nn.init.kaiming_uniform_(x.weight, a=math.sqrt(5))


def _lora_b_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA B weight to zeros.
    """
    nn.init.zeros_(x.weight)
