# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
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
