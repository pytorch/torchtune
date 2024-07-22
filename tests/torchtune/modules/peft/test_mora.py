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
from torchtune import utils
from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook
from torchtune.modules.peft import MoRALinear
from torchtune.utils.seed import set_seed

RANK = 4
ALPHA = 1.0
BSZ = 2
SEQ_LEN = 32
EXPECTED_VAL = 9.394


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestMoRALinear:
    """
    Class for testing our MoRALinear implementation.
    """

    @pytest.fixture
    def in_dim(self) -> int:
        return 64

    @pytest.fixture
    def out_dim(self) -> int:
        return 64

    @pytest.fixture
    def inputs(self, in_dim) -> torch.Tensor:
        inputs = torch.randn(BSZ, SEQ_LEN, in_dim)
        return inputs

    @pytest.fixture
    def mora_linear(self, in_dim, out_dim) -> MoRALinear:
        mora_linear = MoRALinear(
            in_dim=in_dim,
            out_dim=out_dim,
            rank=RANK,
            alpha=ALPHA,
            use_bias=True,
        )
        fixed_init_model(mora_linear)
        return mora_linear

    @pytest.fixture
    def qmora_linear(self, in_dim, out_dim) -> MoRALinear:
        with utils.set_default_dtype(torch.bfloat16):
            qmora_linear = MoRALinear(
                in_dim=512,
                out_dim=512,
                rank=RANK,
                alpha=ALPHA,
                use_bias=False,
                quantize_base=True,
            )
            fixed_init_model(qmora_linear, dtype=torch.bfloat16)
            return qmora_linear

    @torch.no_grad()
    def set_dummy_weights_for_merge(self, mora_module):
        mora_module.lora_a.weight = nn.Parameter(
            torch.zeros_like(mora_module.lora_a.weight)
        )
        # mora_module.mora_b.weight = nn.Parameter(
        #     torch.zeros_like(mora_module.mora_b.weight)
        # )
        mora_module.weight = nn.Parameter(torch.zeros_like(mora_module.weight))
        mora_module.bias = nn.Parameter(torch.zeros_like(mora_module.bias))

        # Hardcode some very specific nonzero values to make verification easy
        mora_module.weight[4, 5] = 1
        mora_module.bias[7] = 2
        mora_module.lora_a.weight[1, 32] = 12

    def test_forward(self, inputs, mora_linear, out_dim) -> None:
        expected = torch.tensor(EXPECTED_VAL)
        actual = mora_linear(inputs)
        assert actual.shape == (BSZ, SEQ_LEN, out_dim)
        torch.testing.assert_close(actual.mean(), expected, atol=1e-4, rtol=1e-6)

    def test_mora_weight_nf4_when_quantized(self, qmora_linear):
        assert isinstance(qmora_linear.weight, NF4Tensor)

    def test_quantize_with_bias_raises(self):
        with pytest.raises(NotImplementedError, match="does not support bias"):
            MoRALinear(
                in_dim=512,
                out_dim=512,
                rank=RANK,
                alpha=ALPHA,
                use_bias=True,
                quantize_base=True,
            )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16]) # [torch.bfloat16, torch.float32])
    def test_qmora_parity(self, dtype):
        with utils.set_default_dtype(dtype):
            qmora_linear = MoRALinear(
                in_dim=512,
                out_dim=512,
                rank=RANK,
                alpha=ALPHA,
                use_bias=False,
                quantize_base=True,
            )
            mora_linear = MoRALinear(
                in_dim=512,
                out_dim=512,
                rank=RANK,
                alpha=ALPHA,
                use_bias=False,
                quantize_base=False,
            )

        # set weight of mora_linear to unquantized weight of qmora_linear and check
        # parity.
        mora_linear.weight.data = qmora_linear.weight.to(dtype)

        # Ensure forward passes are the same. This is because MoRALinear should use a special
        # quantized linear operator that runs compute in higher prec (but only saves the 4 bit quantized tensor)
        # for autograd.
        inputs = torch.randn(BSZ, SEQ_LEN, 512, dtype=dtype)
        mora_linear_out = mora_linear(inputs)
        qmora_linear_out = qmora_linear(inputs)
        torch.testing.assert_close(mora_linear_out, qmora_linear_out)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_quantized_state_dict(self, dtype):
        with utils.set_default_dtype(dtype):
            mora_linear = MoRALinear(
                in_dim=512,
                out_dim=512,
                rank=RANK,
                alpha=ALPHA,
                use_bias=False,
                quantize_base=True,
            )

        mora_linear._register_state_dict_hook(
            partial(
                reparametrize_as_dtype_state_dict_post_hook,
                dtype=dtype,
                offload_to_cpu=False,
            )
        )
        sd = mora_linear.state_dict()
        # No nf4 tensors, all have type dtype
        for v in sd.values():
            assert v.dtype == dtype
            assert not isinstance(v, NF4Tensor)

        # Load back in results in re-quant and creates the same nf4 tensor.
        # This also ensures that MoRALinear can load a bf16 state_dict.
        mora_linear_reload = MoRALinear(
            in_dim=512,
            out_dim=512,
            rank=RANK,
            alpha=ALPHA,
            use_bias=False,
            quantize_base=True,
        )
        # Zero out weight to verify reloading works
        mora_linear_reload.weight = nn.Parameter(
            to_nf4(
                torch.zeros_like(
                    mora_linear.weight.get_original_weight(),
                    dtype=dtype,
                    device=mora_linear.weight.device,
                )
            )
        )
        # nf4 tensors should be different
        assert not torch.allclose(
            mora_linear.weight.quantized_data, mora_linear_reload.weight.quantized_data
        )
        # but should be the same after loading
        mora_linear_reload.load_state_dict(sd)
        assert torch.allclose(
            mora_linear.weight.quantized_data, mora_linear_reload.weight.quantized_data
        )
