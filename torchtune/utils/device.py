# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

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
        torch.cuda.set_device(device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def get_device(name: Optional[str] = None) -> torch.device:
    """Function that gets the torch.device based on the input string.

    This currently supports only CPU and GPU devices. If CUDA is available, this function also sets the CUDA device.

    Args:
        name (Optional[str]): The name of the device to use.

    Raises:
        ValueError: If the device is not supported.

    Returns:
        device
    """
    device = torch.device(name) if name is not None else _get_device_from_env()
    if device.type == "cuda" and device.index is None:
        device = _get_device_from_env()

    if name is not None and device.type != device_type:
        raise RuntimeError(
            f"Device type is specified to {name} but got {device.type} from env"
        )

    local_rank = int(os.environ.get("LOCAL_RANK", None))
    if device.type == "cuda" and device.index != local_rank:
        raise RuntimeError(
            f"You can't specify a device index when using distributed training. \
            Device specified is {name} but was assigned cuda:{local_rank}"
        )
    return device
