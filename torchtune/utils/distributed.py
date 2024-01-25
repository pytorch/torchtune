# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from typing import Dict, Optional, Set, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from torchtune.utils.device import _validate_device_from_env, get_device
from torchtune.utils.logging import get_logger

_log: logging.Logger = get_logger()


def _get_sharding_strategy(strategy: str) -> ShardingStrategy:
    """Helper function to convert sharding strategy strings to ShardingStrategy enum."""
    return getattr(ShardingStrategy, strategy)


def _is_distributed() -> bool:
    """Check if all environment variables required to initialize torch.distributed are set
    and distributed is properly installed. This indicates a distributed run.
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    """
    port = os.environ.get("MASTER_PORT", "")
    addr = os.environ.get("MASTER_ADDR", "")
    size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", -1))
    avlb = dist.is_available()
    return bool(port and addr and size >= 1 and rank >= 0 and avlb)


def _broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcasts a tensor from a source to all other processes.

    Args:
        tensor (torch.Tensor): Tensor to broadcast.
        src (int, optional): Source rank. Defaults to 0.

    Returns:
        torch.Tensor: Broadcasted tensor.
    """
    if dist.is_available() and dist.is_initialized():
        device = tensor.device
        if dist.get_backend() == "nccl":
            tensor = tensor.to(get_device("cuda"))
        dist.broadcast(tensor, src=src, group=None)
        return tensor.to(device)
    else:
        return tensor


def init_distributed(**kwargs: Dict) -> bool:  # noqa: DOC106, DOC109
    """Initialize torch.distributed.

    Args:
        **kwargs (Dict): Additional arguments to pass to torch.distributed.init_process_group.

    Returns:
        bool: True if torch.distributed is initialized.

    Raises:
        RuntimeError: If torch.distributed is already initialized.
    """
    if _is_distributed():
        if dist.is_initialized():
            raise RuntimeError("torch.distributed already initialized.")
        dist.init_process_group(**kwargs)
        return True
    else:
        return False


def get_world_size_and_rank() -> Tuple[int, int]:
    """Function that gets the current world size (aka total number
    of ranks) and rank number of the current trainer.

    Returns:
        Tuple[int, int]: world size, rank
    """
    if dist.is_available() and dist.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    else:
        return 1, 0


def get_fsdp(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    strategy: Optional[str] = None,
    auto_wrap_policy: Optional[Set[nn.Module]] = None,
    cpu_offload: bool = False,
    **kwargs
) -> nn.Module:
    """Utility to setup distributed training using the torch.distributed FullyShardedDataParallel (FSDP) module.
    FSDP allows three primary types of data parallel training (these can be set under "strategy"):

        NO_SHARD: No sharding is done, this is standard Data Parallel training. The is typically fastest if the entire
            model and optimizer can fit on a single GPU and you just want to split the batch across ranks.
        SHARD_GRAD_OP: Only gradients and optimizer are sharded across all ranks. This is typically fastest when the
            model can fit on your GPU but there isn't enough room for a forward and backward pass.
        FULL_SHARD: All parameters are sharded across all ranks. This is necessary when even the model cannot fit on a
            single GPU.

    If using sharding, you need to define how the model is sharded. The auto_wrap_policy is a list of model layers
    and blocks that FSDP will use as shards.

    Args:
        model (nn.Module): Model to wrap for distributed training.
        device (torch.device): Device for host model.
        dtype (torch.dtype): dtype used to determine if mixed precision training is used.
        strategy (Optional[str]): Sharding strategy to use. The main options are (FULL_SHARD, SHARD_GRAD_OP, NO_SHARD)
        auto_wrap_policy (Optional[Set[nn.Module]]): Set of model blocks to shard for sharding.
        cpu_offload (bool): Whether to offload sharded parameters to CPU.
        **kwargs: additional arguments to pass to FSDP for distributed training.

    Returns:
        nn.Module: Model wrapped for distributed training

    Raises:
        RuntimeError: If environment not setup for distributed training.
    """
    if dist.is_available() and dist.is_initialized():
        if strategy is None:
            strategy = "NO_SHARD"
        _validate_device_from_env(device)
        wrap_policy = ModuleWrapPolicy(auto_wrap_policy or set())
        mp = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        if cpu_offload:
            _log.warning("CPU offload will significantly reduce performance. Use with caution.")
        return FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            device_id=device,
            param_init_fn=lambda m: m.to_empty(device=device, recurse=False),
            mixed_precision=mp,
            sharding_strategy=_get_sharding_strategy(strategy),
            cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
            **kwargs
        )
    else:
        raise RuntimeError(
            "Distributed environment is not setup. Please run init_distributed() first."
        )
