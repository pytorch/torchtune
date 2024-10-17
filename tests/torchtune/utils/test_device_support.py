#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest

import torch
from torchtune.utils._device_support import (
    DeviceSupport,
    get_device_support,
    get_torch_device,
)


class TestDevice:

    cuda_available: bool = torch.cuda.is_available()

    @pytest.mark.skipif(not cuda_available, reason="The test requires GPUs to run.")
    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_available(self, mock_cuda):
        # Test if CUDA is available, get_device_support should return DeviceSupport.CUDA
        device_support = get_device_support()
        assert device_support == DeviceSupport.CUDA
        assert device_support.device_type == "cuda"
        assert device_support.device_name == "GPU"
        assert device_support.device_backend == "nccl"

    @pytest.mark.skipif(not cuda_available, reason="The test requires GPUs to run.")
    @patch("torch.cuda.is_available", return_value=True)
    def test_get_torch_device_for_cuda(self, mock_cuda):
        # Test if get_torch_device returns the correct torch.cuda module
        torch_device = get_torch_device("cuda")
        assert torch_device == torch.cuda
