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
from torchtune.utils._device import (
    _get_device_type_from_env,
    _setup_cuda_device,
    batch_to_device,
    get_device,
)


class TestDevice:

    cuda_available: bool = torch.cuda.is_available()

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_cpu_device(self, mock_cuda):
        devices = [None, "cpu", "meta"]
        expected_devices = [
            torch.device("cpu"),
            torch.device("cpu"),
            torch.device("meta"),
        ]
        for device, expected_device in zip(devices, expected_devices):
            device = get_device(device)
            assert device == expected_device
            assert device.index is None

    def test_batch_to_device(self):
        batch = {
            "a": torch.ones(1),
            "b": {
                "c": torch.ones(1),
                "d": torch.ones(1),
            },
        }
        device = torch.device("meta")
        batch_to_device(batch, device)
        assert batch["a"].device == device
        assert batch["b"]["c"].device == device
        assert batch["b"]["d"].device == device

        batch["e"] = 0
        with pytest.raises(ValueError):
            batch_to_device(batch, device)

    @pytest.mark.skipif(not cuda_available, reason="The test requires GPUs to run.")
    def test_get_gpu_device(self) -> None:
        device_idx = torch.cuda.device_count() - 1
        assert device_idx >= 0
        with mock.patch.dict(os.environ, {"LOCAL_RANK": str(device_idx)}, clear=True):
            device = get_device()
            assert device.type == "cuda"
            assert device.index == device_idx
            assert device.index == torch.cuda.current_device()

            # Test that we raise an error if the device index is specified on distributed runs
            if device_idx > 0:
                with pytest.raises(
                    RuntimeError,
                    match=f"Device specified is cuda:0 but was assigned cuda:{device_idx}",
                ):
                    device = get_device("cuda:0")

        invalid_device_idx = device_idx + 10
        with mock.patch.dict(os.environ, {"LOCAL_RANK": str(invalid_device_idx)}):
            with pytest.raises(
                RuntimeError,
                match="The local rank is larger than the number of available GPUs",
            ):
                device = get_device("cuda")

        # Test that we fall back to 0 if LOCAL_RANK is not specified
        device = torch.device(_get_device_type_from_env())
        device = _setup_cuda_device(device)
        assert device.type == "cuda"
        assert device.index == 0
        assert device.index == torch.cuda.current_device()
