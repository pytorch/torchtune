# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Set

import torch

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


def set_activation_checkpointing(
    model: nn.Module, auto_wrap_policy: Optional[Set[nn.Module]] = None, **kwargs
) -> None:
    """Utility to setup activation checkpointing and wrap the model for checkpointing.

    Args:
        model (nn.Module): Model to setup activation checkpointing.
        auto_wrap_policy (Optional[Set[nn.Module]]): Policy to wrap module.
        **kwargs: additional arguments to pass to torch.distributed activation checkpointing.
    """
    wrap_policy = ModuleWrapPolicy(auto_wrap_policy or set())
    apply_activation_checkpointing(model, auto_wrap_policy=wrap_policy, **kwargs)


def memory_stats_log(
    prefix: str, device: torch.device, reset_stats: bool = True
) -> None:
    """
    Print a memory summary for the passed in device. If ``reset_stats`` is ``True``, this will
    also reset CUDA's peak memory tracking. This is useful to get data around relative use of peak
    memory (i.e. peak memory during model init, during forward, etc) and optimize memory for
    individual sections of training.

    Args:
        prefix (str): Prefix to prepend to the printed summary.
        device (torch.device): Device to get memory summary for. Only CUDA devices are supported.
        reset_stats (bool): Whether to reset CUDA's peak memory tracking.

    Returns:
        None
    """
    if device.type != "cuda":
        return
    peak_memory_active = torch.cuda.memory_stats().get("active_bytes.all.peak", 0) / 1e9
    peak_mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
    peak_mem_reserved = torch.cuda.max_memory_reserved(device) / 1e9

    ret = f"""
    {prefix}:
    GPU peak memory allocation: {peak_mem_alloc:.2f} GB
    GPU peak memory reserved: {peak_mem_reserved:.2f} GB
    GPU peak memory active: {peak_memory_active:.2f} GB
    """

    if reset_stats:
        torch.cuda.reset_peak_memory_stats(device)

    return ret
