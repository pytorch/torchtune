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

RANK = 4
ALPHA = 1.0
BSZ = 2
SEQ_LEN = 32


@pytest.fixture(autouse=True)
def random():
    seed(16)


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
        )
        fixed_init_model(lora_linear)
        return lora_linear

    def test_forward(self, inputs, lora_linear, out_dim) -> None:
        expected = torch.tensor(1.1252)
        actual = lora_linear(inputs)
        assert_expected(actual.shape, (BSZ, SEQ_LEN, out_dim))
        assert_expected(actual.mean(), expected, atol=1e-4, rtol=1e-6)


class TestLoRAFusedLinear:
    """
    Class for testing our LoRAFusedLinear implementation. Expected values are computed
    from the reference implementation and calculated in scripts/compare_fused_lora.py.
    """

    def get_inputs(self, in_dim):
        inputs = torch.randn(BSZ, SEQ_LEN, in_dim)
        return inputs

    def get_lora_fused_linear(self, in_dim, out_dims, apply_lora):
        lora_fused_linear = LoRAFusedLinear(
            in_dim=in_dim,
            out_dims=out_dims,
            apply_lora=apply_lora,
            rank=RANK,
            alpha=ALPHA,
        )
        fixed_init_model(lora_fused_linear)
        return lora_fused_linear

    @pytest.fixture
    def toy_lora_fused_linear(self):
        in_dim = 2
        out_dims = [1, 3, 5, 6]
        apply_lora = [True, False, False, True]
        toy_lora_fused_linear = self.get_lora_fused_linear(in_dim, out_dims, apply_lora)
        return toy_lora_fused_linear

    def test_lora_invalid_inputs(self):
        with pytest.raises(
            ValueError,
            match="Must have same number of output dims",
        ):
            _ = self.get_lora_fused_linear(
                in_dim=2, out_dims=[4, 5, 6], apply_lora=[True, False]
            )

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
    ):
        inputs = self.get_inputs(in_dim)
        lora_fused_linear = self.get_lora_fused_linear(in_dim, out_dims, apply_lora)
        actual = lora_fused_linear(inputs)
        assert_expected(actual.shape, (BSZ, SEQ_LEN, sum(out_dims)))
        assert_expected(actual.mean(), expected, atol=1e-4, rtol=1e-6)
