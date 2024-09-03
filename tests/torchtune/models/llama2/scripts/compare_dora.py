# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest

import torch
import torch.nn.functional as F
from torch import nn
from torchao.dtypes.nf4tensor import linear_nf4, to_nf4
from torchtune import training
from torchtune.modules.peft import (
    DoRALinear,
    get_merged_lora_ckpt,
    load_dora_magnitudes,
    LoRALinear,
)
from torchtune.training.seed import set_seed


def compare_dora(self, dtype, use_bias, quantize_base):
    dropout = 0.0
    batch_size = 2
    in_dim = 64
    out_dim = 128
    rank = 4
    alpha = 1.0
    use_bias = False
    quantize_base = False
    dtype = torch.bfloat16

    constructor_kwargs = {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "use_bias": use_bias,
        "quantize_base": quantize_base,
    }

    # this combo is not supported yet
    if use_bias:
        with pytest.raises(
            NotImplementedError, match="DoRALinear does not support using bias"
        ):
            DoRALinear(**constructor_kwargs)
        return

    # build our DoRA module and a reference module for comparison
    with training.set_default_dtype(dtype):
        module = DoRALinear(**constructor_kwargs)
        ref = _DoraReference(dtype=dtype, **constructor_kwargs)
        lora_module = LoRALinear(**constructor_kwargs)

    # make the initial parameters equal
    state_dict = ref.state_dict()
    lora_state_dict = ref.state_dict()
    if quantize_base:
        state_dict["weight"] = state_dict["weight"].to(torch.float32)
        lora_state_dict["weight"] = lora_state_dict["weight"].to(torch.float32)
    state_dict["magnitude"] = state_dict.pop("lora_magnitude")
    lora_state_dict.pop("lora_magnitude")
    module.load_state_dict(state_dict)
    lora_module.load_state_dict(lora_state_dict)

    # freeze the base params
    module.weight.requires_grad_(False)
    lora_module.weight.requires_grad_(False)
    ref.weight.requires_grad_(False)
    if use_bias:
        module.bias.requires_grad_(False)
        module.lora_module.requires_grad_(False)
        ref.bias.requires_grad_(False)

    @torch.no_grad
    def _dora_is_the_same_as_lora():
        module.eval()
        lora_module.eval()
        x = torch.randn(batch_size, in_dim, dtype=dtype)
        lora_out = lora_module(x)
        dora_out = module(x)
        return torch.equal(lora_out, dora_out)

    # DoRA initializes the magnitude vector (after the base params are loaded)
    # such that its outputs are initially identical to standard LoRA's outputs.
    # Verify that this is true.
    assert not _dora_is_the_same_as_lora()
    module.initialize_dora_magnitude()
    load_dora_magnitudes(module)
    assert _dora_is_the_same_as_lora()

    def _compare_params():
        assert torch.allclose(
            module.weight.to(torch.float32), ref.weight.to(torch.float32)
        )
        assert torch.allclose(module.lora_a.weight, ref.lora_a.weight)
        assert torch.allclose(module.lora_b.weight, ref.lora_b.weight)
        assert torch.allclose(module.magnitude, ref.lora_magnitude)

    # verify that the param values match the reference
    ref.initialize_dora()
    _compare_params()

    # compare a single training step to the reference
    module.train()
    ref.train()
    opt = torch.optim.Adam(module.parameters())
    opt_ref = torch.optim.Adam(ref.parameters())
    opt.zero_grad()
    opt_ref.zero_grad()
    # x = torch.randn(batch_size, in_dim, dtype=dtype)
    x = torch.randn(batch_size, 32, in_dim)
    y = torch.randn(batch_size, out_dim)
    torch.manual_seed(0)
    y1 = module(x.detach())
    torch.manual_seed(0)
    y2 = ref(x.detach())
    F.mse_loss(y1.to(torch.float32), y.detach()).backward()
    F.mse_loss(y2.to(torch.float32), y.detach()).backward()
    assert torch.allclose(y1, y2)
    assert torch.allclose(module.magnitude.grad, ref.lora_magnitude.grad)
    assert torch.allclose(module.lora_a.weight.grad, ref.lora_a.weight.grad)
    assert torch.allclose(module.lora_b.weight.grad, ref.lora_b.weight.grad)
    opt.step()
    opt_ref.step()
    _compare_params()

    # verify that the merged and unmerged DoRA modules have nearly identical outputs
    state_dict = get_merged_lora_ckpt(_Wrapper(module).state_dict(), rank, alpha)
    merged = _Wrapper(nn.Linear(in_dim, out_dim, bias=use_bias, dtype=dtype))
    merged.load_state_dict(state_dict)
    merged = merged.layer
    module.eval()
    merged.eval()
    with torch.no_grad():
        x = torch.randn(batch_size, in_dim, dtype=dtype)
        y1 = module(x)
        y2 = merged(x)
        mse = F.mse_loss(y1.float(), y2.float())
        assert mse < (1e-8 if dtype == torch.float32 else 1e-2)


class _Wrapper(nn.Module):
    """
    For testing the merged checkpoint which requires that the LoRA layer has a parent.
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)


class _DoraReference(nn.Module):
    """
    DoRA linear layer reference.

    Paper: https://arxiv.org/abs/2402.09353

    Based on the code from:
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/dora.py

    For more info, see the discussion here:
    https://github.com/huggingface/peft/pull/1474
    """

    def __init__(
        self,
        dtype: torch.dtype,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
        quantize_base: bool = False,
        use_dora: bool = True,
    ):
        super().__init__()
        self.use_bias = use_bias
        self.quantize_base = quantize_base
        self.use_dora = use_dora

        linear = nn.Linear(
            in_features=in_dim, out_features=out_dim, bias=use_bias, dtype=dtype
        )
        weight = linear.weight if not quantize_base else to_nf4(linear.weight)
        bias = None
        if use_bias:
            if quantize_base:
                raise NotImplementedError()
            bias = linear.bias
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias) if bias is not None else None
        )

        self.lora_a = nn.Linear(in_dim, rank, bias=False, dtype=dtype)
        self.lora_b = nn.Linear(rank, out_dim, bias=False, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.scaling = alpha / rank
        if use_dora:
            self.lora_magnitude = nn.Parameter(torch.randn(out_dim, dtype=dtype))
        self.dropout = nn.Dropout(p=dropout)

    def initialize_dora(self):
        weight = self.weight.to(self.lora_a.weight.dtype)
        lora_weight = self.lora_b.weight @ self.lora_a.weight
        weight_norm = self._get_weight_norm(weight, lora_weight)
        self.lora_magnitude = nn.Parameter(weight_norm, requires_grad=True)

    def forward(self, x):
        result = self._base_forward(x)
        torch_result_dtype = result.dtype
        x = x.to(self.lora_a.weight.dtype)
        if not self.use_dora:
            result = result + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        else:
            x = self.dropout(x)
            result = result + self._dora_forward(x)
        result = result.to(torch_result_dtype)
        print("result mean", result.mean())
        return result

    def _base_forward(self, x):
        if self.quantize_base:
            return linear_nf4(input=x, weight=self.weight)
        return F.linear(x, self.weight, self.bias)

    def _dora_forward(self, x):
        lora_result = self.lora_b(self.lora_a(x))
        x_eye = torch.eye(
            self.lora_a.weight.shape[1], device=self.lora_a.weight.device, dtype=x.dtype
        )
        lora_weight = self.lora_b(self.lora_a(x_eye)).T

        magnitude = self.lora_magnitude
        weight = self.weight.to(x.dtype)
        weight_norm = self._get_weight_norm(weight, lora_weight.detach())
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        result_dora = (mag_norm_scale - 1) * (
            F.linear(x, weight)
        ) + mag_norm_scale * lora_result * self.scaling
        return result_dora

    def _get_weight_norm(self, weight, lora_weight):
        weight = weight + self.scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm
