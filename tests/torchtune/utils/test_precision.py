# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest
import torch

from torchtune.utils.precision import _get_grad_scaler

from tests.test_utils import assert_expected


class TestGetGradScaler:
    def test_get_grad_scaler(self):
        """
        Tests that the correct gradient scaler is returned based on device and precision.
        """

        for precision in [None, "bf16"]:
            assert_expected(_get_grad_scaler(precision=precision), None)

        assert isinstance(_get_grad_scaler("fp16"), torch.cuda.amp.GradScaler)

        with pytest.raises(ValueError):
            _get_grad_scaler("foo")
