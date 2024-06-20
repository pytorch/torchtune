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
from torchtune.modules.peft import AdapterModule


class LoRALinear(nn.Module, AdapterModule):
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
        use_dora (bool): Decompose the weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
            Default: False
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
        use_dora: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self._quantize_base = quantize_base
        self.use_dora = use_dora

        # 'self.disabled' is a flag showing whether to turn off LoRA adapters,
        # this can be used in DPO for treating the lora adapters as the policy model
        # and disabling it to treat the base model as the reference model
        self.disabled = False

        weight, bias = self._create_weight_and_bias()
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias) if bias is not None else None
        )

        self.lora = LoRA(in_dim, out_dim, rank, alpha, dropout, use_dora)

        # Note: FSDP's meta device initialization contract assumes that a module's
        # reset_parameters method only initializes its own parameters (i.e. no child
        # params are initialized, as is done in initialize_parameters below).
        # For that reason, we patch reset_parameters directly on lora_a and lora_b submodules
        # when using meta device. This is done in
        # torchtune.utils.prepare_model_for_fsdp_with_meta_device.
        # See this issue for more details: https://github.com/pytorch/pytorch/issues/104187.
        # Without meta device, we only need the following:
    #     self.initialize_parameters()

    # def initialize_parameters(self):
    #     # Initialize as in
    #     # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
    #     _lora_a_init_params(self.lora_a)
    #     _lora_b_init_params(self.lora_b)

    #     if self.use_dora:
    #         print("check1")
    #         # NOTE: This initialization is just a fallback. The magnitude is initialized
    #         # after loading the base model weights in `on_base_params_loaded`.
    #         nn.init.ones_(self.lora_magnitude)

    def on_base_params_loaded(self):
        """
        Initialization that occurs after the base model's parameters have been loaded.
        """
        if self.use_dora:
            self.lora.initialize_dora(self.weight)

    def _create_weight_and_bias(self):
        """
        Creates a linear weight and bias tensor, using NF4 dtype if we're quantizing
        (indicated via quantize_base=True).
        """
        in_dim, out_dim, use_bias = self.in_dim, self.out_dim, self.use_bias
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=use_bias)
        weight = linear.weight if not self._quantize_base else to_nf4(linear.weight)
        bias = None
        if self.use_bias:
            if self._quantize_base:
                raise NotImplementedError(
                    "Quantized LoRALinear does not support bias at the moment."
                )
            bias = linear.bias
        return weight, bias

    def adapter_params(self) -> List[str]:
        """
        Return LoRA's parameters as state dict keys relative to the current module.
        """
        # NOTE: this function has to be updated if the attribute names change.
        adapter_params = ["lora.a.weight", "lora.b.weight"]
        if self.use_dora:
            adapter_params.append("lora.magnitude")
        return adapter_params

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``
        """
        base_out = self._base_forward(x)
        if self.disabled:
            return base_out
        lora_out = self.lora(x, self.weight)
        return base_out + lora_out
    
    def _base_forward(self, x):
        if self._quantize_base:
            return linear_nf4(input=x, weight=self.weight)
        return F.linear(x, self.weight, self.bias)


class LoRA(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_dora: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scaling = alpha / rank
        self.use_dora = use_dora

        self.dropout = nn.Dropout(p=dropout)
        self.a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        if self.use_dora:
            print("check0")
            self.magnitude = nn.Parameter(
                torch.empty(1, out_dim, dtype=torch.get_default_dtype())
            )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

        if self.use_dora:
            print("check1")
            # NOTE: This initialization is just a fallback. The magnitude is initialized
            # after loading the base model weights in `on_base_params_loaded`.
            nn.init.ones_(self.magnitude)

    def initialize_dora(self, base_weight):
        """
        DoRA initializes the magnitude vector such that its outputs are initially
        identical to standard LoRA's outputs
        """
        base_weight = base_weight.to(self.a.weight.dtype)
        lora_weight = self.b.weight @ self.a.weight
        weight = base_weight + self.scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1)
        self.magnitude.data = weight_norm
        print("a", self.a.weight.dtype, self.magnitude.dtype)

    def forward(self, x, base_weight):
        x = self.dropout(x)
        out = self.scaling * self.b(self.a(x))
        if self.use_dora:
            out = self._dora_forward(x, base_weight, out)
        return out

    def _dora_forward(self, x, base_weight, lora_out):
        base_weight = base_weight.to(x.dtype)
        lora_weight = self.b.weight @ self.a.weight
        weight = base_weight + self.scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).detach()
        mag_norm_scale = (self.magnitude / weight_norm).view(1, -1)
        base_out = F.linear(x, base_weight)
        dora_out = (mag_norm_scale - 1) * base_out + mag_norm_scale * lora_out
        return dora_out


    
