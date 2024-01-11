# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from datetime import timedelta
from typing import Optional, Union

import torch
from torch.distributed.constants import default_pg_timeout


_log: logging.Logger = logging.getLogger(__name__)


def _check_dist_env() -> bool:
    """
    Check if all environment variables required to initialize torch.distributed are set
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    """
    env_required = (
        os.environ.get("MASTER_PORT"),
        os.environ.get("MASTER_ADDR"),
        os.environ.get("WORLD_SIZE"),
        os.environ.get("RANK"),
    )
    return all(env is not None for env in env_required)


def init_distributed(
    device: Union[str, torch.device],
    *,
    pg_backend: Optional[str] = None,
    pg_timeout: timedelta = default_pg_timeout,
) -> torch.device:
    """Utility function that initializes the device and process group, if applicable.

    The global process group is initialized only if:
        - torch.distributed is available is not already initialized
        - the program has been launched on multiple processes

    This is intended as a convenience to include at the beginning of scripts that follow
    a SPMD-style execution model.


    Args:
        device (Union[str, torch.device]): Device type to initialize.
        pg_backend (Optional[str], optional): The process group backend to use. If None, it will use the
                                    default process group backend from the device
        pg_timeout (timedelta, optional): Timeout for operations executed against the process
                                          group. Default value equals 30 minutes

    Returns:
        The current device.
    """
    if _check_dist_env():
        if not torch.distributed.is_available():
            _log.warning(
                "torch.distributed is not available. Skipping initializing the process group."
            )
            return
        if torch.distributed.is_initialized():
            _log.warning(
                "torch.distributed is already initialized. Skipping initializing the process group."
            )
            return
        pg_backend = (
            pg_backend
            if pg_backend is not None
            else _get_process_group_backend_from_device(device)
        )
        torch.distributed.init_process_group(backend=pg_backend, timeout=pg_timeout)


def _get_process_group_backend_from_device(device: torch.device) -> str:
    """Function that gets the default process group backend from the device."""
    return "nccl" if device.type == "cuda" else "gloo"
