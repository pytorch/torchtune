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


def memory_stats_log(msg: str) -> str:
    return f"""
    Memory Stats {msg}:
    Memory Allocated: {torch.cuda.memory_allocated(device=torch.cuda.current_device()) / 1000**3:.2f} GB
    Memory Reserved: {torch.cuda.memory_reserved(device=torch.cuda.current_device()) / 1000**3:.2f} GB
    Peak Memory: {torch.cuda.max_memory_allocated(device=torch.cuda.current_device()) / 1000**3:.2f} GB
    """
