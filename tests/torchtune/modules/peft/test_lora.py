# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from torchtune.modules.peft import LoRALinear
from torchtune.modules.peft.lora import reset_lora_params
from torchtune.utils.seed import set_seed

from tests.test_utils import assert_expected, fixed_init_model

RANK = 4
ALPHA = 1.0
BSZ = 2
SEQ_LEN = 32


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestLoRAUtils:
    def test_reset_lora_params(self) -> None:
        with torch.device("meta"):
            lora_linear = LoRALinear(
                in_dim=64,
                out_dim=128,
                rank=RANK,
                alpha=ALPHA,
                use_bias=True,
            )

        # _lora_params_initialized should be False
        assert not lora_linear._lora_params_initialized
        # lora_a, lora_b should be on meta device
        assert lora_linear.lora_a.weight.is_meta
        assert lora_linear.lora_b.weight.is_meta
        init_device = torch.device("cpu")
        reset_lora_params(lora_linear, device=init_device)
        assert lora_linear._lora_params_initialized
        assert lora_linear.lora_a.weight.device == init_device
        assert lora_linear.lora_b.weight.device == init_device


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

    def test_forward(self, inputs, lora_linear, out_dim) -> None:
        expected = torch.tensor(1.1252)
        actual = lora_linear(inputs)
        assert_expected(actual.shape, (BSZ, SEQ_LEN, out_dim))
        assert_expected(actual.mean(), expected, atol=1e-4, rtol=1e-6)

    def test_forward_without_lora_reset(self, inputs):
        with torch.device("meta"):
            lora_linear = LoRALinear(
                in_dim=in_dim,
                out_dim=out_dim,
                rank=RANK,
                alpha=ALPHA,
                use_bias=True,
            )

        with pytest.raises(RuntimeError, match="lora reset_lora_params"):
            _ = lora_linear(out)

    def test_lora_meta_init_matches_device(self, in_dim, out_dim, inputs):
        # TODO (rohan-varma): This test is quite limited and it should really test exact parity
        # between the meta and CPU init of LoRALinear. However, this is tricky due to nondeterminsism of
        # kaiming_uniform_. Torch 2.2 adds a generator to this API, once all workloads are on torch 2.2
        # we can add this testing (or add this testing now and gate it on torch 2.2)
        with torch.device("meta"):
            lora_linear_meta = LoRALinear(
                in_dim=in_dim,
                out_dim=out_dim,
                rank=RANK,
                alpha=ALPHA,
                use_bias=True,
            )
        lora_linear_meta.to_empty(device=torch.device("cpu"), recurse=False)
        lora_linear_meta.reset_lora_parameters()
        assert not lora_linear_meta.lora_a.weight.is_meta
        assert not lora_linear_meta.lora_b.weight.is_meta
