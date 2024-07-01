# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from torchtune.modules.layer_norm import LayerNorm
from torchtune.utils.seed import set_seed


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
    def layer_norm(self, dim, eps) -> LayerNorm:
        return LayerNorm(dim, eps=eps)

    def test_forward_fp16(self, layer_norm, input_random_fp16, eps, dim) -> None:
        output_fp16 = layer_norm(input_random_fp16)
        assert (
            output_fp16.dtype == torch.float32
        ), "Expected output to be fp32, but got {output_fp16.dtype=}"
        assert output_fp16.mean() == torch.nn.LayerNorm(
            8, eps=eps
        ), f"Expected {torch.nn.LayerNorm(dim, eps=eps)=}, but got {output_fp16.mean()=}"
