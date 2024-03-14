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
from torchtune.modules.low_precision import _register_nf4_dispatch_ops, FrozenNF4Linear
from torchtune.modules.peft.peft_utils import AdapterModule
from torchtune.utils.tensor_utils import _copy_tensor


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
        use_bias_in_lora_matrices (bool): whether to add biases to the LoRA matrices
            A and B. Default: False
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
        use_bias_in_lora_matrices: bool = False,
        quantize_base: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.alpha = alpha
        self.out_dim = out_dim
        self.use_bias = use_bias
        self._quantize_base = quantize_base
        weight, bias = self._create_weight_and_bias()
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter("bias", nn.Parameter(bias) if bias is not None else None)
        self.dropout = nn.Dropout(p=dropout)
        self.use_bias_in_lora_matrices = use_bias_in_lora_matrices
        self.lora_a = nn.Linear(
            in_features=in_dim, out_features=rank, bias=self.use_bias_in_lora_matrices
        )
        self.lora_b = nn.Linear(
            in_features=rank, out_features=out_dim, bias=self.use_bias_in_lora_matrices
        )
        self.merged = False
        # Note: FSDP's meta device initialization contract assumes that a module's
        # reset_parameters method only initializes its own parameters (i.e. no child
        # params are initialized, as is done in initialize_parameters below).
        # For that reason, we patch reset_parameters directly on lora_a and lora_b submodules
        # when using meta device. This is done in
        # torchtune.utils.distributed.prepare_model_for_fsdp_with_meta_device.
        # See this issue for more details: https://github.com/pytorch/pytorch/issues/104187.
        # Without meta device, we only need the following:
        self.initialize_parameters()
        self.register_state_dict_pre_hook(self._dequant_state_dict_pre_hook)

    def initialize_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        _lora_a_init_params(self.lora_a)
        _lora_b_init_params(self.lora_b)

    def _create_weight_and_bias(self):
        in_dim, out_dim, use_bias = self.in_dim, self.out_dim, self.use_bias
        linear = (
            nn.Linear(in_features=in_dim, out_features=out_dim, bias=use_bias)
            if not self._quantize_base
            else FrozenNF4Linear(in_dim, out_dim, bias=False)
        )
        weight_tensor = (
            linear.weight
            if not self._quantize_base
            else to_nf4(linear.weight.get_original_weight())
        )
        bias = None
        if self.use_bias:
            if self._quantize_base:
                raise NotImplementedError(
                    "Quantized LoRALinear does not support bias at the moment."
                )
            bias = _copy_tensor(linear.bias)
        return weight_tensor, bias

    def _dequant_state_dict_pre_hook(self, *args, **kwargs):
        """
        Pre-hook that converts quantized LoRA weight to bf16 to support
        checkpoints in bf16. TODO (rohan-varma): make this configurable and possibly
        generalize to other dtypes.
        """
        if self._quantize_base:
            self.weight = self.weight.to(torch.bfloat16)

    def adapter_params(self) -> List[str]:
        """
        Return lora_a.weight and lora_b.weight as adapter params.
        If bias is enabled, also return lora_a.bias and lora_b.bias.
        """
        # NOTE: this function has to be updated if the names of "lora_a" and "lora_b"
        # in this module change.
        adapter_params = ["lora_a.weight", "lora_b.weight"]
        if self.use_bias_in_lora_matrices:
            adapter_params.extend(["lora_a.bias", "lora_b.bias"])
        return adapter_params

    def _unquantize_base_weight(self, *args, **kwargs):
        if not self._quantize_base:
            raise RuntimeError("Cannot call _unquantize_base_weight, weights are not quantized")
        # TODO (rohan-varma): only supports bf16 for the time being.
        self.weight = self.weight.to(torch.bfloat16)

    def _quantize_base_weight(self, *args, **kwargs):
        if not self._quantize_base:
            raise RuntimeError("Cannot call _quantize_base_weight, weights are not quantized")
        self.weight = to_nf4(self.weight)

    @property
    def _lora_delta(self):
        return (self.alpha / self.rank) * (self.lora_b.weight @ self.lora_a.weight)

    @torch.no_grad
    def _merge_lora_weights(self, *args, **kwargs):
        if self.merged:
            raise RuntimeError("Cannot call _merge_lora_weights, weights are merged")
        if self.use_bias_in_lora_matrices:
            raise RuntimeError(
                "Cannot merge LoRA weights when LoRA matrices have biases"
            )
        self.weight += self._lora_delta
        self.cached_lora_a_weight = torch.clone(self.lora_a.weight)
        self.cached_lora_b_weight = torch.clone(self.lora_b.weight)
        del self.lora_a
        del self.lora_b
        self.merged = True

    @torch.no_grad
    def _unmerge_lora_weights(self, *args, **kwargs):
        if not self.merged:
            raise RuntimeError(
                "Cannot call _unmerge_lora_weights, weights are not merged"
            )
        self.lora_a = nn.Linear(self.in_dim, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, self.out_dim, bias=False)
        self.lora_a.weight = nn.Parameter(self.cached_lora_a_weight)
        self.lora_b.weight = nn.Parameter(self.cached_lora_b_weight)
        del self.cached_lora_a_weight
        del self.cached_lora_b_weight
        self.weight -= self._lora_delta
        self.merged = False

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``

        """
        if self._quantize_base:
            out = linear_nf4(input=x, weight=self.weight)
        else:
            out = F.linear(x, self.weight, self.bias)
        lora_out = self.lora_a(self.dropout(x))
        lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)
        return out + lora_out


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
