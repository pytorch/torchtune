# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest
import torch
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from torchtune.utils.precision import (
    get_autocast_manager,
    get_grad_scaler,
    get_supported_dtypes,
)

from tests.test_utils import assert_expected


class TestPrecisionUtils:
    def test_get_grad_scaler(self):
        """
        Tests that the correct gradient scaler is returned based on precision.
        """

        for precision in [None, "bf16"]:
            assert_expected(get_grad_scaler(precision=precision, fsdp=False), None)
            assert_expected(get_grad_scaler(precision=precision, fsdp=True), None)

        assert isinstance(
            get_grad_scaler("fp16", fsdp=False), torch.cuda.amp.GradScaler
        )
        assert isinstance(get_grad_scaler("fp16", fsdp=True), ShardedGradScaler)

        with pytest.raises(ValueError):
            get_grad_scaler("foo", fsdp=False)

    def test_get_autocast_manager(self):
        """
        Tests that the correct autocast manager is returned based on precision.
        """

        for precision in ["fp16", "bf16", "fp32", None]:
            assert isinstance(
                get_autocast_manager(device_type="cuda", precision=precision),
                torch.autocast,
            )

    def test_get_supported_dtyes(self):
        assert set(get_supported_dtypes()) == {"fp16", "bf16", "fp32"}
