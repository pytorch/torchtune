# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest

import torch
import torch.nn.functional as F
from tests.test_utils import fixed_init_model
from torch import nn
from torchao.dtypes.nf4tensor import NF4Tensor, to_nf4
from torchtune import training
from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook
from torchtune.modules.peft import LoRALinear
from torchtune.modules.peft.peft_utils import (
    get_merged_lora_ckpt,
    notify_base_params_loaded,
)
from torchtune.utils.seed import set_seed

RANK = 4
ALPHA = 1.0
BSZ = 2
SEQ_LEN = 32
EXPECTED_VAL = 1.1252


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestLoRALinear:
    """
    Class for testing our LoRALinear implementation. Expected values are computed
    from the reference implementation and calculated in scripts/compare_lora.py.
    """

    @pytest.fixture
    def in_dim(self) -> int:
        return 64

    @pytest.fixture
    def out_dim(self) -> int:
        return 128

    @pytest.fixture
    def inputs(self, in_dim) -> torch.Tensor:
        inputs = torch.randn(BSZ, SEQ_LEN, in_dim)
        return inputs

    @pytest.fixture
    def lora_linear(self, in_dim, out_dim) -> LoRALinear:
        lora_linear = LoRALinear(
            in_dim=in_dim,
            out_dim=out_dim,
            rank=RANK,
            alpha=ALPHA,
            use_bias=True,
        )
        fixed_init_model(lora_linear)
        return lora_linear

    @pytest.fixture
    def qlora_linear(self, in_dim, out_dim) -> LoRALinear:
        with training.set_default_dtype(torch.bfloat16):
            qlora_linear = LoRALinear(
                in_dim=512,
                out_dim=512,
                rank=RANK,
                alpha=ALPHA,
                use_bias=False,
                quantize_base=True,
            )
            fixed_init_model(qlora_linear, dtype=torch.bfloat16)
            return qlora_linear

    @torch.no_grad()
    def set_dummy_weights_for_merge(self, lora_module):
        lora_module.lora_a.weight = nn.Parameter(
            torch.zeros_like(lora_module.lora_a.weight)
        )
        lora_module.lora_b.weight = nn.Parameter(
            torch.zeros_like(lora_module.lora_b.weight)
        )
        lora_module.weight = nn.Parameter(torch.zeros_like(lora_module.weight))
        lora_module.bias = nn.Parameter(torch.zeros_like(lora_module.bias))

        # Hardcode some very specific nonzero values to make verification easy
        lora_module.weight[4, 5] = 1
        lora_module.bias[7] = 2
        lora_module.lora_a.weight[1, 25] = 3
        lora_module.lora_b.weight[32, 1] = 12

    def test_forward(self, inputs, lora_linear, out_dim) -> None:
        expected = torch.tensor(EXPECTED_VAL)
        actual = lora_linear(inputs)
        assert actual.shape == (BSZ, SEQ_LEN, out_dim)
        torch.testing.assert_close(actual.mean(), expected, atol=1e-4, rtol=1e-6)

    def test_lora_weight_nf4_when_quantized(self, qlora_linear):
        assert isinstance(qlora_linear.weight, NF4Tensor)

    def test_quantize_with_bias_raises(self):
        with pytest.raises(NotImplementedError, match="does not support bias"):
            LoRALinear(
                in_dim=512,
                out_dim=512,
                rank=RANK,
                alpha=ALPHA,
                use_bias=True,
                quantize_base=True,
            )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_qlora_parity(self, dtype):
        with training.set_default_dtype(dtype):
            qlora_linear = LoRALinear(
                in_dim=512,
                out_dim=512,
                rank=RANK,
                alpha=ALPHA,
                use_bias=False,
                quantize_base=True,
            )
            lora_linear = LoRALinear(
                in_dim=512,
                out_dim=512,
                rank=RANK,
                alpha=ALPHA,
                use_bias=False,
                quantize_base=False,
            )

        # set weight of lora_linear to unquantized weight of qlora_linear and check
        # parity.
        lora_linear.weight.data = qlora_linear.weight.to(dtype)

        # Ensure forward passes are the same. This is because LoRALinear should use a special
        # quantized linear operator that runs compute in higher prec (but only saves the 4 bit quantized tensor)
        # for autograd.
        inputs = torch.randn(BSZ, SEQ_LEN, 512, dtype=dtype)
        lora_linear_out = lora_linear(inputs)
        qlora_linear_out = qlora_linear(inputs)
        torch.testing.assert_close(lora_linear_out, qlora_linear_out)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_quantized_state_dict(self, dtype):
        with training.set_default_dtype(dtype):
            lora_linear = LoRALinear(
                in_dim=512,
                out_dim=512,
                rank=RANK,
                alpha=ALPHA,
                use_bias=False,
                quantize_base=True,
            )

        lora_linear._register_state_dict_hook(
            partial(
                reparametrize_as_dtype_state_dict_post_hook,
                dtype=dtype,
                offload_to_cpu=False,
            )
        )
        sd = lora_linear.state_dict()
        # No nf4 tensors, all have type dtype
        for v in sd.values():
            assert v.dtype == dtype
            assert not isinstance(v, NF4Tensor)

        # Load back in results in re-quant and creates the same nf4 tensor.
        # This also ensures that LoRALinear can load a bf16 state_dict.
        lora_linear_reload = LoRALinear(
            in_dim=512,
            out_dim=512,
            rank=RANK,
            alpha=ALPHA,
            use_bias=False,
            quantize_base=True,
        )
        # Zero out weight to verify reloading works
        lora_linear_reload.weight = nn.Parameter(
            to_nf4(
                torch.zeros_like(
                    lora_linear.weight.get_original_weight(),
                    dtype=dtype,
                    device=lora_linear.weight.device,
                )
            )
        )
        # nf4 tensors should be different
        assert not torch.allclose(
            lora_linear.weight.quantized_data, lora_linear_reload.weight.quantized_data
        )
        # but should be the same after loading
        lora_linear_reload.load_state_dict(sd)
        assert torch.allclose(
            lora_linear.weight.quantized_data, lora_linear_reload.weight.quantized_data
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("dropout", [0.0, 0.1])
    @pytest.mark.parametrize("use_bias", [False, True])
    @pytest.mark.parametrize("quantize_base", [False, True])
    def test_dora(self, dtype, dropout, use_bias, quantize_base):
        batch_size = 2
        in_dim = 256
        out_dim = 256
        rank = 2
        alpha = 1.0

        constructor_kwargs = {
            "in_dim": in_dim,
            "out_dim": out_dim,
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "use_bias": use_bias,
            "quantize_base": quantize_base,
            "use_dora": True,
        }

        # this combo is not supported yet
        if use_bias and quantize_base:
            with pytest.raises(NotImplementedError, match="does not support bias"):
                LoRALinear(**constructor_kwargs)
            return

        # build our LoRA module and a reference module for comparison
        with utils.set_default_dtype(dtype):
            module = LoRALinear(**constructor_kwargs)
            ref = _DoraReference(dtype=dtype, **constructor_kwargs)

        # make the initial parameters equal
        # with torch.no_grad():
        #     ref.weight.data = module.weight.data.clone().to(dtype)
        #     if use_bias:
        #         ref.bias.data = module.bias.data.clone()
        #     ref.lora_a.weight.data = module.lora.a.weight.data.clone()
        #     ref.lora_b.weight.data = module.lora.b.weight.data.clone()
        #     ref.lora_magnitude.data = module.lora.magnitude.data.clone()
        state_dict = ref.state_dict()
        if quantize_base:
            state_dict["weight"] = state_dict["weight"].to(torch.float32)
        state_dict["lora.a.weight"] = state_dict.pop("lora_a.weight")
        state_dict["lora.b.weight"] = state_dict.pop("lora_b.weight")
        state_dict["lora.magnitude"] = state_dict.pop("lora_magnitude")
        module.load_state_dict(state_dict)

        # freeze the base params
        module.weight.requires_grad_(False)
        ref.weight.requires_grad_(False)
        if use_bias:
            module.bias.requires_grad_(False)
            ref.bias.requires_grad_(False)

        @torch.no_grad
        def _dora_is_the_same_as_lora():
            module.eval()
            x = torch.randn(batch_size, in_dim, dtype=dtype)
            module.use_dora = False
            lora_out = module(x)
            module.use_dora = True
            dora_out = module(x)
            return torch.allclose(lora_out, dora_out)

        # DoRA initializes the magnitude vector (after the base params are loaded)
        # such that its outputs are initially identical to standard LoRA's outputs.
        # Verify that this is true.
        # assert not _dora_is_the_same_as_lora()
        notify_base_params_loaded(module)
        assert _dora_is_the_same_as_lora()

        def _compare_params():
            assert torch.equal(
                module.weight.to(torch.float32), ref.weight.to(torch.float32)
            )
            if use_bias:
                assert torch.equal(module.bias, ref.bias)
            assert torch.equal(module.lora.a.weight, ref.lora_a.weight)
            assert torch.equal(module.lora.b.weight, ref.lora_b.weight)
            assert torch.equal(module.lora.magnitude, ref.lora_magnitude)

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
        x = torch.randn(batch_size, in_dim, dtype=dtype)
        y = torch.randn(batch_size, out_dim)
        torch.manual_seed(0)
        y1 = module(x.detach())
        torch.manual_seed(0)
        y2 = ref(x.detach())
        F.mse_loss(y1.to(torch.float32), y.detach()).backward()
        F.mse_loss(y2.to(torch.float32), y.detach()).backward()
        assert torch.equal(y1, y2)
        assert torch.equal(module.lora.magnitude.grad, ref.lora_magnitude.grad)
        assert torch.equal(module.lora.a.weight.grad, ref.lora_a.weight.grad)
        assert torch.equal(module.lora.b.weight.grad, ref.lora_b.weight.grad)
        opt.step()
        opt_ref.step()
        _compare_params()

        # verify that the merged and unmerged DoRA modules have identical outputs
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
            assert torch.allclose(y1, y2, atol=1e-6 if dtype == torch.float32 else 1e-2)


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
        use_dora: bool = False,
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
        self.scaling = alpha / rank
        if use_dora:
            self.lora_magnitude = nn.Parameter(torch.empty(1, out_dim, dtype=dtype))
        self.dropout = nn.Dropout(p=dropout)

    def initialize_dora(self):
        print("b", self.weight.dtype, self.lora_a.weight.dtype)
        weight = self.weight.to(self.lora_a.weight.dtype)
        lora_weight = self.lora_b.weight @ self.lora_a.weight
        weight_norm = self._get_weight_norm(weight, lora_weight)
        self.lora_magnitude = nn.Parameter(weight_norm, requires_grad=True)

    def initialize_dora2(self):
        lora_a_weight = self.lora_a.weight
        lora_b_weight = self.lora_b.weight
        dtype_is_fp16 = lora_a_weight.dtype == torch.float16
        if dtype_is_fp16:
            lora_a_weight = lora_a_weight.float()
            lora_b_weight = lora_b_weight.float()
        weight = self.weight.to(torch.float32)
        lora_weight = lora_b_weight @ lora_a_weight
        if dtype_is_fp16:
            lora_weight = lora_weight.half()
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
