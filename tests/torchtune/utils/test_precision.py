# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import contextlib

import pytest
import torch

from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from torchtune.utils.precision import (
    _set_float32_precision,
    autocast,
    get_dtype,
    get_gradient_autoscaler,
    list_dtypes,
)

from tests.test_utils import assert_expected


class TestPrecisionUtils:

    cuda_available: bool = torch.cuda.is_available()

    def test_get_dtype(self):
        """
        Tests that the correct dtype is returned based on the input string.
        """
        dtypes = [None, torch.half] + list_dtypes()
        expected_dtypes = [
            torch.float32,
            torch.float16,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float32,
            torch.float64,
        ]
        for dtype, expected_dtype in zip(dtypes, expected_dtypes):
            assert (
                get_dtype(dtype) == expected_dtype
            ), f"{dtype} should return {expected_dtype}"

    def test_grad_autoscaler(self):
        """
        Tests that the correct gradient scaler is returned based on precision.
        """
        for dtype in [None, "bf16"]:
            assert_expected(
                get_gradient_autoscaler(dtype=dtype, fsdp=False)._enabled, False
            )
            assert_expected(
                get_gradient_autoscaler(dtype=dtype, fsdp=True)._enabled, False
            )

        assert isinstance(
            get_gradient_autoscaler("fp16", fsdp=False), torch.cuda.amp.GradScaler
        )
        assert isinstance(get_gradient_autoscaler("fp16", fsdp=True), ShardedGradScaler)

        with pytest.raises(ValueError):
            get_gradient_autoscaler("foo", fsdp=False)

    def test_autocast(self):
        """
        Tests that the correct autocast manager is returned based on precision.
        """

        for dtype in ["fp16"]:
            assert isinstance(
                autocast(device="cpu", dtype=dtype),
                torch.autocast,
            )
        for dtype in ["fp32", None]:
            assert isinstance(
                autocast(device="cpu", dtype=dtype),
                contextlib.nullcontext,
            )

    def test_list_dtyes(self):
        assert set(list_dtypes()) == {"fp16", "bf16", "fp32", "tf32", "fp64"}

    @pytest.mark.skipif(not cuda_available, reason="The test requires GPUs to run.")
    def test_set_float32_precision(self) -> None:
        _set_float32_precision("highest")
        assert torch.get_float32_matmul_precision() == "highest"
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32

        _set_float32_precision("high")
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cudnn.allow_tf32
        assert torch.backends.cuda.matmul.allow_tf32
