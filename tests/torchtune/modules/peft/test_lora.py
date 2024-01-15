# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from torchtune.modules.peft.lora import LoRAFusedLinear, LoRALinear
from torchtune.utils.env import seed

from tests.test_utils import assert_expected, fixed_init_model


@pytest.fixture(autouse=True)
def random():
    seed(16)


@pytest.fixture
def rank() -> int:
    return 4


@pytest.fixture
def alpha() -> float:
    return 1.0


@pytest.fixture
def bsz() -> int:
    return 2


@pytest.fixture
def seq_len() -> int:
    return 32


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
    def inputs(self, bsz, seq_len, in_dim) -> torch.Tensor:
        inputs = torch.randn(bsz, seq_len, in_dim)
        return inputs

    @pytest.fixture
    def lora_linear(self, in_dim, out_dim, rank, alpha) -> LoRALinear:
        lora_linear = LoRALinear(
            in_dim=in_dim,
            out_dim=out_dim,
            rank=rank,
            alpha=alpha,
        )
        fixed_init_model(lora_linear)
        return lora_linear

    def test_forward(self, inputs, lora_linear, bsz, seq_len, out_dim) -> None:
        expected = torch.tensor(1.1252)
        actual = lora_linear(inputs)
        assert_expected(actual.shape, (bsz, seq_len, out_dim))
        assert_expected(actual.mean(), expected, atol=1e-4, rtol=1e-6)


class TestLoRAFusedLinear:
    """
    Class for testing our LoRAFusedLinear implementation. Expected values are computed
    from the reference implementation and calculated in scripts/compare_fused_lora.py.
    """

    @pytest.fixture
    def get_inputs(self, bsz, seq_len, in_dim):
        def create_inputs(in_dim):
            inputs = torch.randn(bsz, seq_len, in_dim)
            return inputs

        return create_inputs

    @pytest.fixture
    def get_lora_fused_linear(self, rank, alpha):
        def create_lora_fused_linear(in_dim, out_dims, apply_lora):
            lora_fused_linear = LoRAFusedLinear(
                in_dim=in_dim,
                out_dims=out_dims,
                apply_lora=apply_lora,
                rank=rank,
                alpha=alpha,
            )
            fixed_init_model(lora_fused_linear)
            return lora_fused_linear

        return create_lora_fused_linear

    @pytest.fixture
    def toy_lora_fused_linear(self, get_lora_fused_linear):
        in_dim = 2
        out_dims = [1, 3, 5, 6]
        apply_lora = [True, False, False, True]
        toy_lora_fused_linear = get_lora_fused_linear(in_dim, out_dims, apply_lora)
        return toy_lora_fused_linear

    def test_get_lora_indices(self, toy_lora_fused_linear):
        expected = [0, 9, 10, 11, 12, 13, 14]
        actual = toy_lora_fused_linear._get_lora_indices()
        assert_expected(actual, expected)

    def test_zero_pad(self, toy_lora_fused_linear):
        x = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14],
            ]
        )
        expected = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7],
                [8, 0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 11, 12, 13, 14],
            ]
        )
        actual = toy_lora_fused_linear._zero_pad(x)
        assert_expected(actual, expected)

    @pytest.mark.parametrize(
        "out_dims, apply_lora, in_dim, expected",
        [
            ([32, 32, 32], [False, False, False], 32, torch.tensor(0.3148)),
            ([32, 32, 32], [True, False, False], 32, torch.tensor(0.2752)),
            ([32, 32, 32], [True, True, True], 32, torch.tensor(0.1899)),
            ([64, 16, 16], [True, False, True], 64, torch.tensor(1.0982)),
        ],
    )
    def test_forward(
        self,
        in_dim,
        out_dims,
        apply_lora,
        expected,
        bsz,
        seq_len,
        get_inputs,
        get_lora_fused_linear,
    ):
        inputs = get_inputs(in_dim)
        lora_fused_linear = get_lora_fused_linear(in_dim, out_dims, apply_lora)
        actual = lora_fused_linear(inputs)
        assert_expected(actual.shape, (bsz, seq_len, sum(out_dims)))
        assert_expected(actual.mean(), expected, atol=1e-4, rtol=1e-6)
