# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Optional, Set, Tuple, Union

import torch
from torch import nn
from torch.distributed.constants import default_pg_timeout
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from torchtune.utils.device import get_device
from torchtune.utils.precision import get_dtype


_log: logging.Logger = logging.getLogger(__name__)


def _get_sharding_strategy(strategy: str) -> ShardingStrategy:
    """Helper function to convert sharding strategy strings to ShardingStrategy enum."""
    return getattr(ShardingStrategy, strategy)


def _is_distributed() -> bool:
    """Check if all environment variables required to initialize torch.distributed are set
    and distributed is properly installed. This indicates a distributed run.
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    """
    env_required = (
        os.environ.get("MASTER_PORT"),
        os.environ.get("MASTER_ADDR"),
        os.environ.get("WORLD_SIZE"),
        os.environ.get("RANK"),
    )
    return (
        all(env is not None for env in env_required)
        and torch.distributed.is_available()
    )


def _broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> None:
    """Broadcasts a tensor from a source to all other processes.

    Args:
        tensor (torch.Tensor): Tensor to broadcast.
        src (int, optional): Source rank. Defaults to 0.

    Returns:
        torch.Tensor: Broadcasted tensor.
    """
    if _is_distributed():
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=None, timeout=default_pg_timeout
            )
        torch.distributed.broadcast(tensor, src=src, group=None)


def get_world_size_and_rank() -> Tuple[int, int]:
    """Function that gets the current world size (aka total number
    of ranks) and rank number of the current trainer.

    Returns:
        Tuple[int, int]: world size, rank
    """
    if _is_distributed():
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=None, timeout=default_pg_timeout
            )
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    else:
        return 1, 0


def get_distributed(
    model: nn.Module,
    device: Union[str, torch.device, None],
    dtype: Union[str, torch.dtype],
    strategy: Optional[str] = None,
    auto_wrap_policy: Optional[Set[nn.Module]] = None,
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
        device (Union[str, torch.device, None]): Device for host model.
        dtype (Union[str, torch.dtype]): dtype used to determine if mixed precision training is used.
        strategy (Optional[str]): Sharding strategy to use. The main options are (FULL_SHARD, SHARD_GRAD_OP, NO_SHARD)
        auto_wrap_policy (Optional[Set[nn.Module]]): Set of model blocks to shard for sharding.
        **kwargs: additional arguments to pass to FSDP for distributed training.

    Returns:
        nn.Module: Model wrapped for distributed training

    Raises:
        RuntimeError: If environment not setup for distributed training.
    """
    if _is_distributed():
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=None, timeout=default_pg_timeout
            )
        if strategy is not None:
            strategy = "NO_SHARD"
        wrap_policy = ModuleWrapPolicy(auto_wrap_policy or set())
        dtype = get_dtype(dtype)
        device = get_device(device)
        mp = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        return FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            device_id=device,
            param_init_fn=lambda m: m.to_empty(device=device, recurse=False),
            mixed_precision=mp,
            sharding_strategy=_get_sharding_strategy(strategy),
            **kwargs
        )
    else:
        raise RuntimeError(
            "Environment not setup for distributed training. Please see documentation on launching a distributed job."
        )
