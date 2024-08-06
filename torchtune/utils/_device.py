# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Optional, cast

import torch
if torch.xpu.is_available():
    import intel_extension_for_pytorch

class _DeviceHandle:
    """
    This is a simple abstraction for computing devices,
    which enables custom backends that implement CUDA-like
    semantics.
    """

    def __init__(self, device: torch.device, backend: Any = None):
        if backend is None:
            try:
                self.__backend = getattr(torch, device.type)
                self.__device = device
            except AttributeError as exc:
                raise AttributeError(
                    f"Device '{device}' does not have a corresponding backend registered as 'torch.{device.type}'."
                ) from exc
        else:
            self.__backend = backend

    @classmethod
    def from_device(cls, device: torch.device) -> "_DeviceHandle":
        """
        Return an device handle corresponding to the device, and through this handle,
        operations with the same semantics as CUDA can be performed on the device.
        Just return torch.cuda if the device is cuda to make attribute-access faster.
        Custom backend must first register a module with the same name with {device.type} on torch.
        """
        if device.type == "cuda":
            return cast(_DeviceHandle, torch.cuda)
        return cls(device)

    def __getattr__(self, __name: str) -> Any:
        try:
            return getattr(self.__backend, __name)
        except AttributeError as exc:
            raise AttributeError(
                f"Custom backend '{self.__device.type}' not implement 'torch.{self.__device.type}.{__name}'"
            ) from exc


def _get_local_rank() -> Optional[int]:
    """Function that gets the local rank from the environment.

    Returns:
        local_rank int or None if not set.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        local_rank = int(local_rank)
    return local_rank


def _setup_device(device: torch.device) -> torch.device:
    """Function that sets the CUDA or XPU device and infers the cuda or xpu
    index if not set.

    Args:
        device (torch.device): The device to set.

    Raises:
        RuntimeError: If device index is not available.

    Returns:
        device
    """
    local_rank = _get_local_rank() or 0
    if device.index is None:
        if torch.xpu.is_available():
            device = torch.device(type="xpu", index=local_rank)
        else:
            device = torch.device(type="cuda", index=local_rank)
    
    print("=====device: ", device)
    # Ensure index is available before setting device
    device_handle = get_device_handle(device)
    if device.index >= device_handle.device_count():
            raise RuntimeError(
                "The local rank is larger than the number of available GPUs."
            )
    device_handle.set_device(device)
    
    return device


def _get_device_type_from_env() -> str:
    """Function that gets the torch.device based on the current machine.

    This currently only supports CPU, CUDA, XPU.

    Returns:
        device
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.xpu.is_available():
        device = 'xpu'
    else:
        device = "cpu"
    return device


def _validate_device_from_env(device: torch.device) -> None:
    """Function that validates the device is correct given the current machine.
    This will raise an error if the device is not available or doesn't match the
    assigned process device on distributed runs.

    Args:
        device (torch.device): The device to validate.

    Raises:
        RuntimeError: If the device is not available or doesn't match the assigned process device.

    Returns:
        device
    """
    local_rank = _get_local_rank()

    # Check if the device index is correct
    if device.type in ["cuda", "xpu"] and local_rank is not None:
        # Ensure device index matches assigned index when distributed training
        if device.index != local_rank:
            raise RuntimeError(
                f"You can't specify a device index when using distributed training. \
                Device specified is {device} but was assigned cuda:{local_rank}"
            )

    # Check if the device is available on this machine
    try:
        torch.empty(0, device=device)
    except RuntimeError as e:
        raise RuntimeError(
            f"The device {device} is not available on this machine."
        ) from e


def get_device(device: Optional[str] = None) -> torch.device:
    """Function that takes an optional device string, verifies it's correct and available given the machine and
    distributed settings, and returns a torch.device. If device string is not provided, this function will
    infer the device based on the environment.

    If CUDA is available and being used, this function also sets the CUDA device.

    Args:
        device (Optional[str]): The name of the device to use.

    Returns:
        torch.device: device.
    """
    if device is None:
        device = _get_device_type_from_env()
        
    device = torch.device(device)
    if device.type == "cuda" or "xpu":
        device = _setup_device(device)
    _validate_device_from_env(device)
    return device

def get_device_handle(device: Optional[str] = None):
    return _DeviceHandle.from_device(device)
