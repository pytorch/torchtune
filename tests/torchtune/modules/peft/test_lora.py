# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest

import torch
from tests.test_utils import fixed_init_model
from torch import nn
from torchao.dtypes.nf4tensor import NF4Tensor, to_nf4
from torchtune import training
from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook
from torchtune.modules.peft import LoRALinear
from torchtune.training.seed import set_seed

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
        def create_lora_linear(use_bias, dtype, in_dim=in_dim, out_dim=out_dim):
            with training.set_default_dtype(dtype):
                lora_linear = LoRALinear(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    rank=RANK,
                    alpha=ALPHA,
                    use_bias=use_bias,
                )
                fixed_init_model(lora_linear)
            return lora_linear

        return create_lora_linear

    @pytest.fixture
    def qlora_linear(self):
        def create_qlora_linear(use_bias, dtype, in_dim=512, out_dim=512):
            with training.set_default_dtype(dtype):
                qlora_linear = LoRALinear(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    rank=RANK,
                    alpha=ALPHA,
                    use_bias=use_bias,
                    quantize_base=True,
                )
                fixed_init_model(qlora_linear)
            return qlora_linear

        return create_qlora_linear

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
        lora_linear = lora_linear(use_bias=True, dtype=torch.float32)
        expected = torch.tensor(EXPECTED_VAL)
        actual = lora_linear(inputs)
        assert actual.shape == (BSZ, SEQ_LEN, out_dim)
        torch.testing.assert_close(actual.mean(), expected, atol=1e-4, rtol=1e-6)

    @pytest.mark.parametrize("use_bias", [True, False])
    def test_lora_weight_nf4_when_quantized(self, use_bias, qlora_linear):
        qlora_linear = qlora_linear(use_bias=use_bias, dtype=torch.bfloat16)
        assert isinstance(qlora_linear.weight, NF4Tensor)
        if use_bias:
            assert not isinstance(qlora_linear.bias, NF4Tensor)
            assert qlora_linear.bias.dtype == torch.bfloat16

    # Note: with bfloat16 F.linear(x, weight, bias) != F.linear(x, weight) + bias.
    # This means we would get different results (irrespective of QLoRA).
    # So we leave that test case out
    @pytest.mark.parametrize(
        "use_bias, dtype",
        [(False, torch.bfloat16), (True, torch.float32), (False, torch.float32)],
    )
    def test_qlora_parity(self, use_bias, dtype, qlora_linear, lora_linear):
        qlora_linear = qlora_linear(
            use_bias=use_bias, dtype=dtype, in_dim=512, out_dim=512
        )
        lora_linear = lora_linear(
            use_bias=use_bias, dtype=dtype, in_dim=512, out_dim=512
        )

        # set weight of lora_linear to unquantized weight of qlora_linear and check
        # parity.
        lora_linear.weight.data = qlora_linear.weight.to(dtype)
        if use_bias:
            lora_linear.bias.data = qlora_linear.bias.detach().clone()
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
