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
            lora_linear.merge_lora_weights()

    def test_merge_lora_weights(self, lora_linear):
        self.set_dummy_weights_for_merge(lora_linear)
        lora_linear.merge_lora_weights()

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
        lora_linear.merge_lora_weights()
        actual = F.linear(inputs, lora_linear.weight, lora_linear.bias)
        torch.testing.assert_close(actual.mean(), expected, atol=1e-4, rtol=1e-6)

    def test_invalid_unmerge_lora_weights_before_merge(self, lora_linear):
        with pytest.raises(RuntimeError, match="weights are not merged"):
            lora_linear.unmerge_lora_weights()

    def test_merge_and_unmerge_lora_weights(self, lora_linear):
        sd_pre = lora_linear.state_dict()
        lora_linear.merge_lora_weights()
        lora_linear.unmerge_lora_weights()
        assert sd_pre.keys() == lora_linear.state_dict().keys()
        for k in sd_pre.keys():
            torch.testing.assert_close(sd_pre[k], lora_linear.state_dict()[k])
