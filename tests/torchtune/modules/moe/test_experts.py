# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, fixed_init_model
from torch import nn
from torchtune.modules.moe import GroupedExperts, LoRAGroupedExperts
from torchtune.modules.peft import LoRALinear
from torchtune.training.seed import set_seed

RANK = 4
ALPHA = 1.0
SEQ_LEN = 32


@pytest.fixture(autouse=True)
def random():
    set_seed(1)


class TestExperts:
    """
    Class for testing our Experts module.
    """

    @pytest.fixture
    def dim(self) -> int:
        return 1280

    @pytest.fixture
    def hidden_dim(self) -> int:
        return 8192

    @pytest.fixture
    def num_experts(self) -> int:
        return 8

    @pytest.fixture
    def num_tokens_per_expert(self, num_experts) -> int:
        return torch.tensor([1, 2, 1, 4, 3, 1, 2, 2], dtype=torch.int)

    @pytest.fixture
    def experts(self, dim, hidden_dim, num_experts) -> nn.Module:
        experts = GroupedExperts(
            dim=dim, hidden_dim=hidden_dim, num_experts=num_experts
        )
        fixed_init_model(experts, min_val=-0.1, max_val=0.1)
        return experts

    @torch.no_grad()
    def test_forward(self, experts, num_tokens_per_expert, dim):
        """
        Test that the forward pass of the experts works as expected.
        """
        x = torch.randn((16, dim))
        out = experts(x, num_tokens_per_expert)

        assert out.shape == (16, dim)
        assert_expected(out.mean().item(), 120.8260, atol=1e-3, rtol=1e-3)


class TestLoRAGroupedExperts:
    @pytest.fixture
    def dim(self) -> int:
        return 64

    @pytest.fixture
    def hidden_dim(self) -> int:
        return 128

    @pytest.fixture
    def num_experts(self) -> int:
        return 8

    @pytest.fixture
    def experts_per_token(self) -> int:
        return 2

    @pytest.fixture
    def num_tokens_per_expert(self, num_experts) -> int:
        return torch.tensor([1, 2, 1, 4, 3, 1, 2, 2], dtype=torch.int)

    @pytest.fixture
    def inputs(self, dim, num_experts, experts_per_token) -> torch.Tensor:
        inputs = torch.randn(num_experts * experts_per_token, SEQ_LEN, dim)
        return inputs

    @pytest.fixture
    def experts(self, dim, hidden_dim, num_experts) -> nn.Module:
        experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
        )
        fixed_init_model(experts, min_val=-0.1, max_val=0.1)
        return experts

    @pytest.fixture
    def lora_experts(self, dim, hidden_dim, num_experts) -> nn.Module:
        experts = LoRAGroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            rank=RANK,
            alpha=ALPHA,
        )
        fixed_init_model(experts, min_val=-0.1, max_val=0.1)
        return experts

    @pytest.fixture
    def lora_linear(self, dim, hidden_dim):
        def create_lora_linear(dim=dim, hidden_dim=hidden_dim):
            lora_linear = LoRALinear(
                in_dim=dim,
                out_dim=hidden_dim,
                rank=RANK,
                alpha=ALPHA,
            )
            fixed_init_model(lora_linear)
            return lora_linear

        return create_lora_linear

    def test_lora_tc_layer_forward(self, lora_linear, lora_experts, inputs):
        """Compare TC forward with LoRALinear as reference"""
        lora = lora_linear()
        actual = lora_experts._lora_tc_layer_forward(
            inputs[0],
            lora.weight.T,
            lora.lora_a.weight.T,
            lora.lora_b.weight.T,
        )
        expected = lora(inputs[0])
        assert_expected(actual, expected, rtol=1e-6, atol=1e-4)

    def test_forward_disabled(
        self, experts, lora_experts, inputs, num_tokens_per_expert
    ):
        """Test forward with lora layers disabled and comparing with GroupedExperts"""
        lora_experts.disabled = True
        actual = lora_experts(inputs, num_tokens_per_expert)
        expected = experts(inputs, num_tokens_per_expert)
        assert_expected(actual, expected, rtol=1e-6, atol=1e-4)

    def test_forward(
        self,
        lora_experts,
        inputs,
        num_experts,
        experts_per_token,
        dim,
        num_tokens_per_expert,
    ) -> None:
        expected = torch.tensor(0.441491)
        actual = lora_experts(inputs, num_tokens_per_expert)
        assert actual.shape == (num_experts * experts_per_token, SEQ_LEN, dim)
        torch.testing.assert_close(actual.mean(), expected, atol=1e-4, rtol=1e-6)
