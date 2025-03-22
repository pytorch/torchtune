# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from tests.test_utils import assert_expected
from torch.nn.functional import normalize

from torchtune.modules.rms_norm import RMSNorm
from torchtune.training.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(0)


class TestRMSNorm:
    """
    Class for testing our RMSNorm implementation. Expected tensors
    are generated using torch.nn.functional.normalization:

    RMSNorm(x) = normalize(x, p=2, dim=-1) * (dim ** 0.5)
    """

    @pytest.fixture
    def dim(self) -> int:
        return 8

    @pytest.fixture
    def eps(self) -> float:
        return 1e-6

    @pytest.fixture
    def input_ones(self, dim) -> torch.Tensor:
        return torch.ones(dim, dtype=torch.float)

    @pytest.fixture
    def input_random(self, dim) -> torch.Tensor:
        return torch.randn(dim, dtype=torch.float)

    @pytest.fixture
    def input_random_fp16(self, dim) -> torch.Tensor:
        return torch.randn(dim, dtype=torch.float16)

    @pytest.fixture
    def rms_norm(self, dim, eps) -> RMSNorm:
        return RMSNorm(dim, eps=eps)

    def test_forward(self, rms_norm, input_ones, input_random, dim) -> None:
        output_ones = rms_norm(input_ones)
        output_random = rms_norm(input_random)

        expected_random = normalize(input_random, p=2, dim=-1) * (dim**0.5)

        assert_expected(output_ones, input_ones)
        assert_expected(output_random, expected_random)

    def test_forward_fp16(self, rms_norm, input_random_fp16, dim) -> None:
        output_fp16 = rms_norm(input_random_fp16)

        # convert input to float since rms_norm computes in fp32
        expected_fp16 = normalize(input_random_fp16.float(), p=2, dim=-1) * (dim**0.5)

        assert_expected(output_fp16, expected_fp16, atol=1e-7, rtol=1e-3)
        assert output_fp16.dtype == torch.float32
