# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import List

import torch.nn.functional as F
import torch

from torch import nn, Tensor

from torchao.dtypes.nf4tensor import linear_nf4, to_nf4
from torchtune.modules.low_precision import _register_nf4_dispatch_ops  # noqa: F401
from torchtune.modules.peft.peft_utils import AdapterModule


class MoRALinear(nn.Module, AdapterModule):
    """MoRA linear layer as introduced in `MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning <https://arxiv.org/abs/2405.12130>`_.

    MoRA replaces the low-rank approximation matrices A and B with a square matrix, resulting in a higher rank trainable layer. In a linear layer instead of
    :math:`x \\mapsto W_0x` a LoRALinear layer is defined as
    :math:`x \\mapsto W_0x + (\\alpha / r)BAx`, where :math:`r` is the rank of
    the matrices :math:`A` and :math:`B` and :math:`\\alpha` is a scaling factor.
    As in the original implementation LoRA, we support dropout before multiplication
    by the MoRA matrix.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        rank (int): low-rank which is converted to high-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability. Default: 0.0
        use_bias (bool): whether to include bias in the original linear layer.
            Default: False
        quantize_base (bool): Whether to quantize base linear weight or not.
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
    ):
        super().__init__()
        self.in_dim = in_dim
        self.rank = math.ceil(math.sqrt((in_dim + out_dim) * rank)) // 2 * 2
        self.alpha = alpha
        self.out_dim = out_dim
        self.use_bias = use_bias
        self._quantize_base = quantize_base
        weight, bias = self._create_weight_and_bias()
        # 'self.disabled' is a flag showing whether to turn off LoRA adapters,
        # this can be used in DPO for treating the lora adapters as the policy model
        # and disabling it to treat the base model as the reference model
        self.disabled = False
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias) if bias is not None else None
        )
        self.dropout = nn.Dropout(p=dropout)
        self.lora_a = nn.Linear(in_features=self.rank, out_features=self.rank, bias=False)
        # self.lora_b = nn.Linear(in_features=self.rank, out_features=self.rank, bias=False)
        self.cos, self.sin = self.precompute_freqs(self.rank)
        self.merged = False
        # Note: FSDP's meta device initialization contract assumes that a module's
        # reset_parameters method only initializes its own parameters (i.e. no child
        # params are initialized, as is done in initialize_parameters below).
        # For that reason, we patch reset_parameters directly on lora_a and lora_b submodules
        # when using meta device. This is done in
        # torchtune.utils.prepare_model_for_fsdp_with_meta_device.
        # See this issue for more details: https://github.com/pytorch/pytorch/issues/104187.
        # Without meta device, we only need the following:
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        _mora_init_params(self.lora_a)
        # _mora_init_params(self.lora_b)

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
                    "Quantized MoRALinear does not support bias at the moment."
                )
            bias = linear.bias
        return weight, bias

    def adapter_params(self) -> List[str]:
        """
        Return lora_a.weight as adapter params.
        If bias is enabled, also return lora_a.bias
        """
        # NOTE: this function has to be updated if the names of "lora_a" and "lora_b"
        # in this module change.
        adapter_params = ["lora_a.weight",]   # "lora_b.weight"]
        return adapter_params

    def precompute_freqs(self, r: int, base: int = 10000):
        in_f = self.in_dim 
        rb1 = in_f // r if in_f % r == 0 else in_f // r + 1
        inv_freq = 1.0 / (base ** (torch.arange(0, r, 2).float() / r))
        t = torch.arange(rb1)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)
        

    def _apply_mora(self, x):
        """
        Taken from the paper author's github (Apache 2 License): 
        https://github.com/kongds/MoRA/blob/0ff64b144e60b54fe7c0ff7b4e76c99c949e923d/peft-mora/src/peft/tuners/lora/layer.py#L229
        """
        r = self.rank
        in_f, out_f = self.in_dim, self.out_dim
        sum_inter = in_f // r
        rb1 = in_f // r if in_f % r == 0 else in_f // r + 1
        if in_f % r != 0:
            pad_size = r - in_f % r
            x = torch.cat([x, x[..., :pad_size]], dim=-1)
            sum_inter += 1
        in_x = x.view(*x.shape[:-1], sum_inter, r)
        if not hasattr(self, 'cos') and not hasattr(self, 'sin'):
            self.cos, self.sin = self.precompute_freqs(r)
        rh_in_x = torch.cat((-in_x[..., r // 2:], in_x[..., :r // 2]), dim=-1)
        in_x = in_x * self.cos + rh_in_x * self.sin
        out_x = self.lora_a(in_x)
        out_x = out_x.view(*x.shape[:-1], -1)[..., :out_f]
        if out_x.shape[-1] < out_f:
            repeat_time = out_f // out_x.shape[-1]
            if out_f % out_x.shape[-1] != 0:
                repeat_time += 1
            out_x = torch.cat([out_x]*repeat_time, dim=-1)[..., :out_f]
        return out_x


    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape ``(..., in_dim)``
        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``

        """
        if self.disabled:
            return x
        if self._quantize_base:
            out = linear_nf4(input=x, weight=self.weight)
        else:
            out = F.linear(x, self.weight, self.bias)

        # out = self.dropout(out)

        return out + self._apply_mora(self.dropout(out))


def _mora_init_params(x: nn.Linear) -> None:
    """
    Initialize MoRA weights to zeros.
    """
    nn.init.zeros_(x.weight)

# possible cleaner implementation of ROPE
# def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
#     freqs = 1.0 / (
#         base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
#     )
#     t = torch.arange(seq_len, device=freqs.device)
#     freqs = torch.outer(t, freqs)
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
#     cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
#     return cache.to(dtype=torch.bfloat16)
