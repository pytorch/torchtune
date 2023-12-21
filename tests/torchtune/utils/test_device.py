#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest import mock
from unittest.mock import patch

import pytest

import torch
from torchtune.utils.device import _get_device_from_env, maybe_enable_tf32


class TestDevice:

    cuda_available: bool = torch.cuda.is_available()

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_cpu_device(self, _) -> None:
        device = _get_device_from_env()
        assert device.type == "cpu"
        assert device.index is None

    @pytest.mark.skipif(not cuda_available, reason="The test requires GPUs to run.")
    def test_get_gpu_device(self) -> None:
        device_idx = torch.cuda.device_count() - 1
        assert device_idx >= 0
        with mock.patch.dict(os.environ, {"LOCAL_RANK": str(device_idx)}, clear=True):
            device = _get_device_from_env()
            assert device.type == "cuda"
            assert device.index == device_idx
            assert device.index == torch.cuda.current_device()

        invalid_device_idx = device_idx + 10
        with mock.patch.dict(os.environ, {"LOCAL_RANK": str(invalid_device_idx)}):
            with pytest.raises(
                RuntimeError,
                match="The local rank is larger than the number of available GPUs",
            ):
                device = _get_device_from_env()

        # Test that we fall back to 0 if LOCAL_RANK is not specified
        device = _get_device_from_env()
        assert device.type == "cuda"
        assert device.index == 0
        assert device.index == torch.cuda.current_device()

    @pytest.mark.skipif(not cuda_available, reason="The test requires GPUs to run.")
    def test_maybe_enable_tf32(self) -> None:
        maybe_enable_tf32("highest")
        assert torch.get_float32_matmul_precision() == "highest"
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32

        maybe_enable_tf32("high")
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cudnn.allow_tf32
        assert torch.backends.cuda.matmul.allow_tf32
