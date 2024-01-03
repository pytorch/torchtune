# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import ContextManager, Dict, Optional, Union

import torch

from torch.cuda.amp import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

_type_str_to_dtype: Dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": None,
}


def _is_unsupported_precision(precision: Optional[str]) -> None:
    if precision is not None and precision not in _type_str_to_dtype.keys():
        raise ValueError(
            f"`precision` must be one of None, {','.join(_type_str_to_dtype.keys())}."
        )


def get_grad_scaler(
    precision: Optional[str], fsdp: bool
) -> Optional[Union[GradScaler, ShardedGradScaler]]:
    """
    Returns a gradient scaler for mixed-precision training.
    Args:
        precision (Optional[str]): Low precision used for CUDA automatic mixed precision.
        fsdp (bool): Whether FSDP is being used for training, in which case a shard-aware gradient scaler is returned.
    Returns:
        Optional[Union[GradScaler, ShardedGradScaler]]: Gradient scaler object if using one of the supported
        precision types, else `None`.
    """
    _is_unsupported_precision(precision)

    if precision == "fp16":
        return GradScaler(enabled=True) if not fsdp else ShardedGradScaler(enabled=True)

    return None


def get_autocast_manager(device_type: str, precision: Optional[str]) -> ContextManager:
    """
    Returns an autocast manager for mixed-precision training.
    Args:
        device_type (str): Device type. Must be 'cpu' or 'cuda'.
        precision (Optional[str]): Low precision used for CUDA automatic mixed precision.
    Returns:
        Generator: Autocast manager object if using fp16 precision, otherwise an instance of
            `contextlib.nullcontext`.
    """

    _is_unsupported_precision(precision)

    return torch.autocast(
        device_type=device_type,
        dtype=_type_str_to_dtype.get(precision, None),
        enabled=(precision is not None),
    )
