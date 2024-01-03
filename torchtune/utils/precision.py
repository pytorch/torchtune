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

_LOW_PRECISION_STR_TYPES: List[str] = ["fp16", "bf16"]
_type_str_to_dtype: Dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _is_unsupported_precision(precision: Optional[str]) -> None:
    if precision is not None and precision not in _LOW_PRECISION_STR_TYPES:
        raise ValueError(
            f"`precision` must be one of None, {','.join(_LOW_PRECISION_STR_TYPES)}."
        )


def _get_grad_scaler(
    precision: Optional[str], fsdp: bool
) -> Optional[Union[GradScaler, ShardedGradScaler]]:
    """
    Returns a gradient scaler for mixed-precision training.
    Args:
        precision (Optional[str]): Low precision used for CUDA automatic mixed precision.
        fsdp: (bool): Whether FSDP is being used for training, in which case a shard-aware gradient scaler is returned.
    Returns:
        Optional[Union[GradScaler, ShardedGradScaler]]: Gradient scaler object if using one of the supported
        precision types, else `None`.

    Raises:
        ValueError: If precision is not one of None, fp16, or bf16.

    """
    if _is_unsupported_precision(precision):
        raise ValueError(
            f"`precision` must be one of None, {','.join(_LOW_PRECISION_STR_TYPES)}."
        )

    if precision == "fp16":
        return GradScaler(enabled=True) if not fsdp else ShardedGradScaler(enabled=True)

    return None


def _get_autocast_manager(device_type: str, precision: Optional[str]) -> ContextManager:
    """
    Returns an autocast manager for mixed-precision training.
    Args:
        device_type (str): Device type. Must be 'cpu' or 'cuda'.
        precision (Optional[str]): Low precision used for CUDA automatic mixed precision.
    Returns:
        Generator: Autocast manager object if using fp16 precision, otherwise an instance of
            `contextlib.nullcontext`.

    Raises:
        ValueError: If precision is not one of None, fp16, or bf16.
    """
    if _is_unsupported_precision(precision):
        raise ValueError(
            f"`precision` must be one of None, {','.join(_LOW_PRECISION_STR_TYPES)}."
        )
    return torch.autocast(
        device_type=device_type,
        dtype=_type_str_to_dtype.get(precision, None),
        enabled=(precision is not None),
    )
