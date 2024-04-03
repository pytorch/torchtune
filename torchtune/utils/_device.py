# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import torch


def _get_local_rank() -> Optional[int]:
    """Function that gets the local rank from the environment.

    Returns:
        local_rank int or None if not set.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        local_rank = int(local_rank)
    return local_rank


def _setup_cuda_device(device: torch.device) -> torch.device:
    """Function that sets the CUDA device and infers the cuda
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
        device = torch.device(type="cuda", index=local_rank)

    # Ensure index is available before setting device
    if device.index >= torch.cuda.device_count():
        raise RuntimeError(
            "The local rank is larger than the number of available GPUs."
        )

    torch.cuda.set_device(device)
    return device


def _get_device_type_from_env() -> str:
    """Function that gets the torch.device based on the current machine.

    This currently only supports CPU, CUDA.

    Returns:
        device
    """
    if torch.cuda.is_available():
        device = "cuda"
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
    if device.type == "cuda" and local_rank is not None:
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
    """Function that takes or device or device string, verifies it's correct and availabe given the machine and
    distributed settings, and returns a torch.device.

    If CUDA is available and being used, this function also sets the CUDA device.

    Args:
        device (Optional[str]): The name of the device to use.

    Returns:
        device
    """
    if device is None:
        device = _get_device_type_from_env()
    device = torch.device(device)
    if device.type == "cuda":
        device = _setup_cuda_device(device)
    _validate_device_from_env(device)
    return device
