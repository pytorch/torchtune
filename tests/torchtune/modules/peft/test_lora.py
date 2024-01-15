# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from torchtune.modules.peft.lora import LoRALinear
from torchtune.utils.seed import set_seed

from tests.test_utils import assert_expected, fixed_init_model


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestLoRALinear:
    """
    Class for testing our LoRALinear implementation. Expected values are computed
    from the reference implementation and calculated in scripts/compare_lora.py.
    """

    @pytest.fixture
    def bsz(self) -> int:
        return 2

    @pytest.fixture
    def seq_len(self) -> int:
        return 32

    @pytest.fixture
    def in_dim(self) -> int:
        return 64

    @pytest.fixture
    def out_dim(self) -> int:
        return 128

    @pytest.fixture
    def rank(self) -> int:
        return 4

    @pytest.fixture
    def alpha(self) -> float:
        return 1.0

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
