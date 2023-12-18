# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import contextlib

import pytest
import torch

from torchtune.utils.precision import _get_autocast_manager, _get_grad_scaler

from tests.test_utils import assert_expected


class TestPrecisionUtils:
    def test_get_grad_scaler(self):
        """
        Tests that the correct gradient scaler is returned based on precision.
        """

        for precision in [None, "bf16"]:
            assert_expected(_get_grad_scaler(precision=precision), None)

        assert isinstance(_get_grad_scaler("fp16"), torch.cuda.amp.GradScaler)

        with pytest.raises(ValueError):
            _get_grad_scaler("foo")

    def test_get_autocast_manager(self):
        """
        Tests that the correct autocast manager is returned based on precision.
        """

        for precision in ["fp16", "bf16"]:
            assert isinstance(
                _get_autocast_manager(device_type="cuda", precision=precision),
                torch.autocast,
            )

        # TODO: can change this to just return nullcontext for fp32 as well
        with pytest.raises(ValueError):
            _get_autocast_manager(device_type="cuda", precision="fp32")

        assert isinstance(_get_autocast_manager("cuda", None), contextlib.nullcontext)
