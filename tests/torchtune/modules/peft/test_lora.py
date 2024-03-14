# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
import torch.nn.functional as F
from tests.test_utils import fixed_init_model
from torch import nn
from torchao.dtypes.nf4tensor import NF4Tensor
from torchtune import utils
from torchtune.modules.peft import LoRALinear
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
        with utils.set_default_dtype(torch.bfloat16):
            qlora_linear = LoRALinear(
                in_dim=512,
                out_dim=512,
                rank=RANK,
                alpha=ALPHA,
                use_bias=False,
                quantize_base=True,
            )
            fixed_init_model(qlora_linear)
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

    def test_invalid_merge_lora_weights_with_bias(self):
        lora_linear = LoRALinear(
            in_dim=64,
            out_dim=128,
            rank=RANK,
            alpha=ALPHA,
            use_bias_in_lora_matrices=True,
        )
        with pytest.raises(RuntimeError, match="LoRA matrices have biases"):
            lora_linear._merge_lora_weights()

    def test_merge_lora_weights(self, lora_linear):
        self.set_dummy_weights_for_merge(lora_linear)
        lora_linear._merge_lora_weights()

        expected_weight = torch.clone(lora_linear.weight)
        expected_weight[4, 5] = 1
        # [alpha (=1) / rank (=4)] * lora_b (=12) * lora_b (=3)
        expected_weight[25, 32] = 9
        expected_bias = torch.clone(lora_linear.bias)

        assert lora_linear.merged
        assert not hasattr(lora_linear, "lora_a")
        assert not hasattr(lora_linear, "lora_b")
        assert isinstance(lora_linear.cached_lora_a_weight, torch.Tensor)
        assert isinstance(lora_linear.cached_lora_b_weight, torch.Tensor)

    @torch.no_grad()
    def test_merge_lora_weights_forward(self, inputs, lora_linear):
        expected = torch.tensor(EXPECTED_VAL)
        lora_linear._merge_lora_weights()
        actual = F.linear(inputs, lora_linear.weight, lora_linear.bias)
        torch.testing.assert_close(actual.mean(), expected, atol=1e-4, rtol=1e-6)

    def test_invalid_unmerge_lora_weights_before_merge(self, lora_linear):
        with pytest.raises(RuntimeError, match="weights are not merged"):
            lora_linear._unmerge_lora_weights()

    def test_merge_and_unmerge_lora_weights(self, lora_linear):
        inputs = torch.randn(BSZ, SEQ_LEN, lora_linear.in_dim)

        out_pre = lora_linear(inputs)
        sd_pre = lora_linear.state_dict()

        lora_linear._merge_lora_weights()
        lora_linear._unmerge_lora_weights()

        out_post = lora_linear(inputs)

        torch.testing.assert_close(out_pre, out_post)
        torch.testing.assert_close(sd_pre, lora_linear.state_dict())

    def test_merge_and_unmerge_qlora_weights(self, qlora_linear):
        inputs = torch.randn(2, 512, dtype=torch.bfloat16)
        pre_merge_out = qlora_linear(inputs)
        sd_pre_merge = qlora_linear.state_dict()
        qlora_linear._merge_lora_weights()
        qlora_linear._unmerge_lora_weights()
        post_unmerge_out = qlora_linear(inputs)
        assert torch.allclose(post_unmerge_out, pre_merge_out)
        sd_post_unmerge = qlora_linear.state_dict()
        w1 = sd_pre_merge["weight"]
        w2 = sd_post_unmerge["weight"]
        assert torch.allclose(w1.quantized_data, w2.quantized_data)

    def test_lora_weight_nf4_when_quantized(self):
        lora_linear = LoRALinear(
            in_dim=in_dim,
            out_dim=out_dim,
            rank=RANK,
            alpha=ALPHA,
            use_bias=False,
            quantize_base=True,
        )
        assert isinstance(lora_linear.weight, NF4Tensor)

    def test_quantize_with_bias_raises(self):
        with pytest.raises(NotImplementedError, match="does not support bias"):
            LoRALinear(
                in_dim=in_dim,
                out_dim=out_dim,
                rank=RANK,
                alpha=ALPHA,
                use_bias=True,
                quantize_base=True,
            )

    def test_quantized_state_dict_bf16(self):
        lora_linear = LoRALinear(
            in_dim=in_dim,
            out_dim=out_dim,
            rank=RANK,
            alpha=ALPHA,
            use_bias=False,
            quantize_base=True,
        )
        sd = lora_linear.state_dict()
        # No nf4 tensors, all bf16
        for v in sd.values():
            assert v.dtype == torch.bfloat16
            assert not isinstance(v, NF4Tensor)

        # Load back in results in re-quant and creates the same nf4 tensor.
        # This also ensures that LoRALinear can load a bf16 state_dict.
        lora_linear_reload = LoRALinear(
            in_dim=in_dim,
            out_dim=out_dim,
            rank=RANK,
            alpha=ALPHA,
            use_bias=False,
            quantize_base=True,
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
