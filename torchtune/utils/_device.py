# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import torch

from torchtune.utils._import_guard import _SUPPORTS_FLEX_ATTENTION

if _SUPPORTS_FLEX_ATTENTION:
    from torch.nn.attention.flex_attention import BlockMask
else:
    BlockMask = torch.Tensor

from torchtune.utils._device_support import (
    get_device_support,
    get_torch_device,
    is_npu_available,
)


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
    """Function that sets the CUDA or NPU device and infers the cuda
    index if not set.

    Args:
        device (torch.device): The device to set.

    Raises:
        RuntimeError: If device index is not available.

    Returns:
        device
    """
    local_rank = _get_local_rank() or 0
    device_type = get_device_support().device_type
    device_name = get_device_support().device_name
    torch_device = get_torch_device()

    if device.index is None:
        device = torch.device(type=device_type, index=local_rank)

    # Ensure index is available before setting device
    if device.index >= torch_device.device_count():
        raise RuntimeError(
            f"The local rank is larger than the number of available {device_name}s."
        )
    torch_device.set_device(device)
    return device


def _get_device_type_from_env() -> str:
    """Function that gets the torch.device based on the current machine.

    This currently only supports CPU, CUDA, NPU.

    Returns:
        device
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif is_npu_available:
        device = "npu"
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
    if device.type in ["cuda", "npu"] and local_rank is not None:
        # Ensure device index matches assigned index when distributed training
        if device.index != local_rank:
            raise RuntimeError(
                f"You can't specify a device index when using distributed training. \
                Device specified is {device} but was assigned cuda/npu:{local_rank}"
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
    distributed settings, and returns a :func:`~torch.device`. If device string is not provided, this function will
    infer the device based on the environment.

    If CUDA or NPU is available and being used, this function also sets the CUDA or NPU device.

    Args:
        device (Optional[str]): The name of the device to use, e.g. "cuda" or "cpu" or "npu".

    Example:
        >>> device = get_device("cuda")
        >>> device
        device(type='cuda', index=0)

    Returns:
        torch.device: Device
    """
    if device is None:
        device = _get_device_type_from_env()
    device = torch.device(device)
    if device.type in ["cuda", "npu"]:
        device = _setup_device(device)
    _validate_device_from_env(device)
    return device


def batch_to_device(batch: dict, device: torch.device) -> None:
    """Function that takes a dictionary (or nested dictionary) of tensors and sets them
    all to the same device. This utility is intended to be used for batches of data to be
    moved to device, the update is inplace.

    Args:
        batch (dict): dict of Tensors or more nested dicts of tensors.
        device (torch.device): torch device to move the tensor's too

    Raises:
        AttributeError: if batch dict contains anything other than tensors
    """
    for k, v in batch.items():
        if isinstance(v, dict):
            batch_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif _SUPPORTS_FLEX_ATTENTION and isinstance(v, BlockMask):
            batch[k] = v.to(device)
        else:
            raise ValueError(
                f"""To use batch_to_device, all elements in the batch must be a dict or Tensor.
Got key "{k}" with value of type {type(v)}"""
            )
