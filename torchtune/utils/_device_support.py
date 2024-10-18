# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from enum import Enum
from typing import Optional

import torch


def is_torch_npu_available() -> bool:
    """Check the availability of NPU"""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


is_npu_available = is_torch_npu_available()


class DeviceSupport(Enum):
    """
    This is a simple enum for Non-CPU compute devices,
    which enables custom backends that implement CUDA-like semantics.
    """

    CUDA = ("cuda", "GPU", "nccl")
    NPU = ("npu", "NPU", "hccl")

    def __init__(self, device_type: str, device_name: str, device_backend: str):
        self.device_type = device_type
        self.device_name = device_name
        self.device_backend = device_backend

    @staticmethod
    def from_type(device_type: str):
        for member in DeviceSupport:
            if member.device_type == device_type:
                return member
        raise ValueError(f"Unknown device type: {device_type}.")


def _get_device_support_from_env() -> DeviceSupport:
    """function that gets the DeviceSupport with Non-CPU compute devices based on the current machine.

    This currently only supports CUDA, NPU.

    Raises:
        RuntimeError: If Non-CPU compute devices is not available.

    Returns:
        device_support: DeviceSupport
    """
    if is_npu_available:
        return DeviceSupport.NPU
    elif torch.cuda.is_available():
        return DeviceSupport.CUDA
    else:
        raise RuntimeError("No available device found.")


def get_device_support(device_type: Optional[str] = None) -> DeviceSupport:
    """Function that takes an optional device string, verifies it's correct and available given the machine and
    distributed settings, and returns a enum:`DeviceSupport`. If device string is not provided, this function will
    infer the device based on the environment.

    Args:
        device_type (Optional[str]): The name of the device to use, e.g. "cuda" or "npu".

    Example:
        >>> device_support = get_device_support("cuda")
        >>> device_support
        device_support(type='cuda', name='GPU')

    Returns:
        device_support: DeviceSupport
    """
    if device_type is not None:
        device_support = DeviceSupport.from_type(device_type)
    else:
        device_support = _get_device_support_from_env()
    return device_support


def get_torch_device(device_type: Optional[str] = None) -> any:
    """Return the corresponding torch attribute based on the device type string.

    Args:
        device_type(Optional[str]): The device type name, e.g., 'cuda', 'npu'.

    Returns:
        module: The corresponding torch module, or None if not found.
    """
    if device_type is None:
        device_type = get_device_support().device_type
    try:
        return getattr(torch, device_type)
    except AttributeError:
        print(f"Device Module '{device_type}' not found in torch.")
        return None
