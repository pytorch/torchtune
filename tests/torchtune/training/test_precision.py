# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


from unittest import mock

import pytest
import torch

from torchtune.training.precision import (
    _set_float32_precision,
    get_dtype,
    PRECISION_STR_TO_DTYPE,
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
        dtypes = [None, torch.half] + list(PRECISION_STR_TO_DTYPE.keys())
        expected_dtypes = [
            torch.float32,
            torch.float16,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ]
        for dtype, expected_dtype in zip(dtypes, expected_dtypes):
            if dtype == "bf16" and not verify_bf16_support():
                continue  # skip bf16 tests if not supported.
            assert (
                get_dtype(dtype) == expected_dtype
            ), f"{dtype} should return {expected_dtype}"

    @mock.patch("torchtune.training.precision.verify_bf16_support", return_value=False)
    def test_error_bf16_unsupported(self, mock_verify):
        """
        Tests that an error is raised if bf16 is specified but not supported.
        """
        with pytest.raises(
            RuntimeError, match="bf16 precision was requested but not available"
        ):
            get_dtype(torch.bfloat16)

    @pytest.mark.skipif(not cuda_available, reason="The test requires GPUs to run.")
    def test_set_float32_precision(self) -> None:
        setattr(  # noqa: B010
            torch.backends, "__allow_nonbracketed_mutation_flag", True
        )
        _set_float32_precision("highest")
        assert torch.get_float32_matmul_precision() == "highest"
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32

        _set_float32_precision("high")
        setattr(  # noqa: B010
            torch.backends, "__allow_nonbracketed_mutation_flag", False
        )
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
            validate_expected_param_dtype(m.named_parameters(), dtype=torch.float16)
