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
from torchtune.modules.peft import DoRALinear
from torchtune.training.seed import set_seed

RANK = 4
ALPHA = 1.0
BSZ = 2
SEQ_LEN = 32
EXPECTED_VAL = 0.05201


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestDoRALinear:
    """
    Class for testing our DoRALinear implementation. Expected values are computed
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
    def dora_linear(self, in_dim, out_dim):
        def create_dora_linear(use_bias, dtype, in_dim=in_dim, out_dim=out_dim):
            with training.set_default_dtype(dtype):
                dora_linear = DoRALinear(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    rank=RANK,
                    alpha=ALPHA,
                    use_bias=use_bias,
                )

                fixed_init_model(dora_linear)
            return dora_linear

        return create_dora_linear

    @pytest.fixture
    def qdora_linear(self):
        def create_qdora_linear(
            use_bias=False, dtype=torch.bfloat16, in_dim=512, out_dim=512
        ):
            with training.set_default_dtype(dtype):
                qdora_linear = DoRALinear(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    rank=RANK,
                    alpha=ALPHA,
                    use_bias=use_bias,
                    quantize_base=True,
                )
                fixed_init_model(qdora_linear)
            return qdora_linear

        return create_qdora_linear

    def test_forward(self, inputs, dora_linear, out_dim) -> None:
        dora_linear = dora_linear(use_bias=False, dtype=torch.float32)
        expected = torch.tensor(EXPECTED_VAL)
        actual = dora_linear(inputs)
        assert actual.shape == (BSZ, SEQ_LEN, out_dim)
        torch.testing.assert_close(actual.mean(), expected, atol=1e-4, rtol=1e-6)

    @pytest.mark.parametrize("use_bias", [True, False])
    def test_dora_weight_nf4_when_quantized(self, use_bias, qdora_linear):
        qdora_linear = qdora_linear(use_bias=use_bias, dtype=torch.bfloat16)
        assert isinstance(qdora_linear.weight, NF4Tensor)
        if use_bias:
            assert not isinstance(qdora_linear.bias, NF4Tensor)
            assert qdora_linear.bias.dtype == torch.bfloat16

    # Note: with bfloat16 F.linear(x, weight, bias) != F.linear(x, weight) + bias.
    # This means we would get different results (irrespective of QDoRA).
    # So we leave that test case out
    @pytest.mark.parametrize(
        "use_bias, dtype",
        [(False, torch.bfloat16), (True, torch.float32), (False, torch.float32)],
    )
    def test_qdora_parity(self, use_bias, dtype, dora_linear, qdora_linear):
        with training.set_default_dtype(dtype):
            qdora_linear = qdora_linear(
                use_bias=use_bias, dtype=dtype, in_dim=512, out_dim=512
            )
            dora_linear = dora_linear(
                use_bias=use_bias, dtype=dtype, in_dim=512, out_dim=512
            )

        # set weight of dora_linear to unquantized weight of qdora_linear and check
        # parity.
        dora_linear.weight.data = qdora_linear.weight.to(dtype)
        if use_bias:
            dora_linear.bias.data = qdora_linear.bias.detach().clone()
        qdora_linear.initialize_dora_magnitude()
        dora_linear.initialize_dora_magnitude()

        # Ensure forward passes are the same. This is because DoRALinear should use a special
        # quantized linear operator that runs compute in higher prec (but only saves the 4 bit quantized tensor)
        # for autograd.
        inputs = torch.randn(BSZ, SEQ_LEN, 512, dtype=dtype)
        torch.manual_seed(0)
        dora_linear_out = dora_linear(inputs)
        torch.manual_seed(0)
        qdora_linear_out = qdora_linear(inputs)
        torch.testing.assert_close(dora_linear_out, qdora_linear_out)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_quantized_state_dict(self, dtype):
        with training.set_default_dtype(dtype):
            dora_linear = DoRALinear(
                in_dim=512,
                out_dim=512,
                rank=RANK,
                alpha=ALPHA,
                use_bias=False,
                quantize_base=True,
            )

        dora_linear._register_state_dict_hook(
            partial(
                reparametrize_as_dtype_state_dict_post_hook,
                dtype=dtype,
                offload_to_cpu=False,
            )
        )
        sd = dora_linear.state_dict()
        # No nf4 tensors, all have type dtype
        for v in sd.values():
            assert v.dtype == dtype
            assert not isinstance(v, NF4Tensor)

        # Load back in results in re-quant and creates the same nf4 tensor.
        # This also ensures that DoRALinear can load a bf16 state_dict.
        dora_linear_reload = DoRALinear(
            in_dim=512,
            out_dim=512,
            rank=RANK,
            alpha=ALPHA,
            use_bias=False,
            quantize_base=True,
        )
        # Zero out weight to verify reloading works
        dora_linear_reload.weight = nn.Parameter(
            to_nf4(
                torch.zeros_like(
                    dora_linear.weight.get_original_weight(),
                    dtype=dtype,
                    device=dora_linear.weight.device,
                )
            )
        )
        # nf4 tensors should be different
        assert not torch.allclose(
            dora_linear.weight.quantized_data, dora_linear_reload.weight.quantized_data
        )
        # but should be the same after loading
        dora_linear_reload.load_state_dict(sd)
        assert torch.allclose(
            dora_linear.weight.quantized_data, dora_linear_reload.weight.quantized_data
        )
