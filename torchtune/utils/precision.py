# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import ContextManager, Dict, List, Optional, Union

import torch

from torch.cuda.amp import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from torchtune.utils.device import _validate_device_from_env

PRECISION_STR_TO_DTYPE: Dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def _set_float32_precision(precision: str = "high") -> None:
    """Sets the precision of float32 matrix multiplications and convolution operations.

    For more information, see the PyTorch docs:
    - https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    - https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.allow_tf32

    Args:
        precision (str): The setting to determine which datatypes to use for matrix multiplication and convolution operations.
    """
    if not torch.cuda.is_available():  # Not relevant for non-CUDA devices
        return
    # set precision for matrix multiplications
    torch.set_float32_matmul_precision(precision)
    # set precision for convolution operations
    if precision == "highest":
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True


def list_dtypes() -> List[str]:
    """
    Return a list of supported dtypes for finetuning.
    """
    return list(PRECISION_STR_TO_DTYPE.keys())


def get_dtype(dtype: Optional[str] = None) -> torch.dtype:
    """Get the torch.dtype corresponding to the given precision string.

    Args:
        dtype (Optional[str]): The precision dtype.

    Raises:
        ValueError: if precision isn't supported by the precision utils

    Returns:
        torch.dtype: The corresponding torch.dtype.
    """
    # None defaults to float32
    if dtype is None:
        return torch.float32

    # Convert to torch.dtype
    dtype = PRECISION_STR_TO_DTYPE.get(dtype, dtype)

    # dtype must be one of the supported precisions
    if dtype not in PRECISION_STR_TO_DTYPE.values():
        raise ValueError(
            f"Dtype {dtype} must be one of {', '.join(list_dtypes())} for finetuning."
        )
    return dtype


def get_gradient_scaler(fsdp: bool = False) -> Union[GradScaler, ShardedGradScaler]:
    """
    Returns a gradient scaler for mixed-precision training.

    Args:
        fsdp (bool): Whether FSDP is being used for training, in which case a shard-aware gradient scaler is returned.

    Returns:
        Union[GradScaler, ShardedGradScaler]: Gradient scaler object
    """
    return GradScaler(enabled=True) if not fsdp else ShardedGradScaler(enabled=True)


def get_autocast(dtype: torch.dtype, device: torch.device) -> ContextManager:
    """
    Intelligently determines, based on the dtype if mixed precision training is supported and
    returns the builtin torch.autocast if applicable.

    Reference: https://pytorch.org/docs/stable/amp.html#torch.autocast

    Args:
        dtype (torch.dtype): dtype used to determine if mixed precision training is used.
        device (torch.device): Pytorch device.
    Returns:
        Autocast manager object if using half precision, otherwise null context
    """
    manager = None
    if dtype in (torch.float16, torch.bfloat16):
        # Note some devices do not support autocasting, and will raise an error.
        _validate_device_from_env(device)
        return torch.autocast(
            device_type=device.type,
            dtype=dtype,
        )
    else:
        return contextlib.nullcontext()
