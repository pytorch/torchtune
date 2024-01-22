# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import socket
from datetime import datetime
from typing import Optional, Set

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


# Should we make this configurable?
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


# TODO: @Evan I'm not sure how you wanna deal with default logger config but just added this for now
logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


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


def stop_record_memory_history() -> None:
    """
    Stops recording the memory history if CUDA is available.
    Logs a message if CUDA is not available or when the recording is stopped.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)


def start_record_memory_history() -> None:
    """
    Starts recording the memory history if CUDA is available.
    The maximum number of memory events per snapshot is defined by MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT.
    Logs a message if CUDA is not available or when the recording is started.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )


def export_memory_snapshot(file_prefix=None) -> None:
    """
    Exports the memory snapshot to a local file if CUDA is available.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return

    if not file_prefix:
        host_name = socket.gethostname()
        timestamp = datetime.now().strftime(TIME_FORMAT_STR)
        file_prefix = f"{host_name}_{timestamp}"

    try:
        logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
        return
