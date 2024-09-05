# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from tests.test_utils import assert_expected

from torchtune.modules.layer_norm import Fp32LayerNorm
from torchtune.training.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(0)


class TestLayerNorm:
    """
    Class for testing our LayerNorm, which is just a wrapper around torch.nn.LayerNorm
    to support fp16 training.
    """

    @pytest.fixture
    def dim(self) -> int:
        return 8

    @pytest.fixture
    def eps(self) -> float:
        return 1e-6

    @pytest.fixture
    def input_random_fp16(self, dim) -> torch.Tensor:
        return torch.randn(dim, dtype=torch.float16)

    @pytest.fixture
    def layer_norm(self, dim, eps) -> Fp32LayerNorm:
        return Fp32LayerNorm(dim, eps=eps)

    def test_forward_fp16(self, layer_norm, input_random_fp16, eps, dim) -> None:
        output_fp16 = layer_norm(input_random_fp16)

        # assert dtype as fp16
        assert (
            output_fp16.dtype == torch.float16
        ), "Expected output to be fp16, but got {output_fp16.dtype=}"

        # assert value as fp32
        expected_output = torch.nn.LayerNorm(dim, eps=eps)(input_random_fp16.float())
        output_fp32 = layer_norm(input_random_fp16.float())
        assert_expected(
            output_fp32.mean(), expected_output.mean(), atol=1e-8, rtol=1e-8
        )
