# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import ContextManager, Dict, List, Optional, Union

import torch

from torch.cuda.amp import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

_precision_str_to_dtype: Dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
    "fp64": torch.float64,
}


def _get_dtype(precision: Optional[Union[str, torch.dtype]] = None) -> torch.dtype:
    """Get the torch.dtype corresponding to the given precision string.

    Args:
        precision (Optional[Union[str, torch.dtype]]): The precision dtype.

    Raises:
        ValueError: if precision isn't supported by the precision utils

    Returns:
        torch.dtype: The corresponding torch.dtype.
    """
    if precision is None:
        return torch.float32
    if type(precision) == torch.dtype:
        return precision
    try:
        if precision == "tf32":
            _set_float32_precision("highest")
        return _precision_str_to_dtype[precision]
    except ValueError as e:
        raise ValueError(
            f"Precision must be one of {','.join(_precision_str_to_dtype.keys())}"
        ) from e


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


def get_supported_dtypes() -> List[str]:
    """
    Get a list of supported precisions to be used with `torch.autocast`.
    """
    return list(_precision_str_to_dtype.keys())


def get_grad_scaler(
    dtype: Optional[Union[str, torch.dtype]], fsdp: bool
) -> Optional[Union[GradScaler, ShardedGradScaler]]:
    """
    Returns a gradient scaler for mixed-precision training.
    Args:
        dtype (Optional[Union[str, torch.dtype]]): dtype used to determine if mixed precision training is used.
        fsdp (bool): Whether FSDP is being used for training, in which case a shard-aware gradient scaler is returned.
    Returns:
        Optional[Union[GradScaler, ShardedGradScaler]]: Gradient scaler object if using one of the supported
        precision types, else `None`.
    """
    dtype = _get_dtype(dtype)

    if precision == torch.float16:
        return GradScaler(enabled=True) if not fsdp else ShardedGradScaler(enabled=True)

    return None


def get_precision_manager(
    device: torch.device, dtype: Optional[Union[str, torch.dtype]]
) -> ContextManager:
    """
    Returns an autocast manager for mixed-precision training.
    Args:
        device (torch.device): Pytorch device.
        dtype (Optional[Union[str, torch.dtype]]): dtype used to determine if mixed precision training is used.
    Returns:
        Generator: Autocast manager object if using half precision, otherwise an instance of
            `contextlib.nullcontext`.
    """
    dtype = _get_dtype(dtype)
    enabled = dtype in (torch.float16, torch.bfloat16)
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
    )
