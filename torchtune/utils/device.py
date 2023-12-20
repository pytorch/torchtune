# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch


def get_device_from_env() -> torch.device:
    """Function that gets the torch.device based on the current environment.

    This currently supports only CPU and GPU devices. If CUDA is available, this function also sets the CUDA device.

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
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def maybe_enable_tf32(precision: str = "high") -> None:
    """Conditionally sets the precision of float32 matrix multiplications and conv operations.

    For more information, see the
    `PyTorch docs <https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html>`_

    Args:
        precision (str): The setting to determine which datatypes to use for matrix multiplication.
    """
    if not torch.cuda.is_available():  # Not relevant for non-CUDA devices
        return
    torch.set_float32_matmul_precision(precision)
    if precision == "highest":
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True


def get_process_group_backend_from_device(device: torch.device) -> str:
    """Function that gets the default process group backend from the device."""
    return "nccl" if device.type == "cuda" else "gloo"
