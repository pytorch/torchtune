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
    get_autocast,
    get_dtype,
    get_gradient_scaler,
    list_dtypes,
    set_default_dtype,
    validate_expected_param_dtype,
    verify_bf16_support,
)


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
            torch.bfloat16 if verify_bf16_support() else torch.float32,
            torch.float32,
            torch.float64,
        ]
        for dtype, expected_dtype in zip(dtypes, expected_dtypes):
            assert (
                get_dtype(dtype) == expected_dtype
            ), f"{dtype} should return {expected_dtype}"

    def test_grad_scaler(self):
        """
        Tests that the correct gradient scaler is returned based on precision.
        """
        assert isinstance(get_gradient_scaler(fsdp=False), torch.cuda.amp.GradScaler)
        assert isinstance(get_gradient_scaler(fsdp=True), ShardedGradScaler)

    def test_autocast(self):
        """
        Tests that the correct autocast manager is returned based on precision.
        """
        device = torch.device("cpu")
        for dtype in [torch.float16]:
            assert isinstance(
                get_autocast(device=device, dtype=dtype),
                torch.autocast,
            )
        for dtype in [torch.float32, torch.float64]:
            assert isinstance(
                get_autocast(device=device, dtype=dtype),
                contextlib.nullcontext,
            )

    def test_list_dtyes(self):
        assert set(list_dtypes()) == {"fp16", "bf16", "fp32", "fp64"}

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

    def test_set_default_dtype(self):
        dtype = torch.bfloat16
        prev_dtype = torch.get_default_dtype()
        with set_default_dtype(dtype):
            assert torch.get_default_dtype() == dtype

        assert torch.get_default_dtype() == prev_dtype

    def test_validate_expected_param_dtype(self):
        """
        Tests that we raise if any model param has a different dtype than the expected dtype.
        """
        m = torch.nn.Linear(10, 10)
        with pytest.raises(ValueError, match=f"has dtype {next(m.parameters()).dtype}"):
            validate_expected_param_dtype(m, dtype=torch.float16)
