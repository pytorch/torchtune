# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional, Union

import torch


def _get_device_from_env() -> torch.device:
    """Function that gets the torch.device based on the current environment.

    This currently supports CPU, GPU and MPS. If CUDA is available, this function also sets the CUDA device.

    Within a distributed context, this function relies on the ``LOCAL_RANK`` environment variable
    to be made available by the program launcher for setting the appropriate device index.

    Raises:
        RuntimeError: If ``LOCAL_RANK`` is outside the range of available GPU devices.

    Returns:
        device
    """
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                "The local rank is larger than the number of available GPUs."
            )
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Function that takes or device or device string, verifies it's correct and availabe given the machine and
    distributed settings, and returns a torch.device.

    If CUDA is available and being used, this function also sets the CUDA device.

    Args:
        device (Optional[Union[str, torch.device]]): The name of the device to use.

    Raises:
        RuntimeError: If the wrong device index is set during distributed training.

    Returns:
        device
    """
    # Convert device string to torch.device
    if type(device) != torch.device:
        if device is None:
            device = _get_device_from_env()
        else:
            device = torch.device(device)

    # Get device rank for cuda devices if not provided, and set Cuda device
    if device.type == "cuda":
        if device.index is None:
            device = _get_device_from_env()
        torch.cuda.set_device(device)

        # Check if the device index is correct when distributed training
        local_rank = os.environ.get("LOCAL_RANK", None)
        if local_rank is not None and device.index != int(local_rank):
            raise RuntimeError(
                f"You can't specify a device index when using distributed training. \
                Device specified is {device} but was assigned cuda:{local_rank}"
            )

    try:
        # Check if the device is available on this machine
        torch.empty(0, device=device)
    except RuntimeError as e:
        raise RuntimeError(
            f"The device {device} is not available on this machine."
        ) from e

    return device
