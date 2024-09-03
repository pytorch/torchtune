# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ._device import get_device
from ._generation import generate, generate_next_token  # noqa

from ._version import torch_version_ge
from .logging import get_logger
from .memory import (  # noqa
    cleanup_before_training,
    create_optim_in_bwd_wrapper,
    get_memory_stats,
    log_memory_stats,
    OptimizerInBackwardWrapper,
    register_optim_in_bwd_hooks,
    set_activation_checkpointing,
)

__all__ = [
    "get_memory_stats",
    "log_memory_stats",
    "get_device",
    "get_logger",
    "set_activation_checkpointing",
    "torch_version_ge",
    "OptimizerInBackwardWrapper",
    "create_optim_in_bwd_wrapper",
    "register_optim_in_bwd_hooks",
    "generate",
    "generate_next_token",
]
