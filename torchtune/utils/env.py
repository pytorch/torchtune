# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
from datetime import timedelta
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.distributed.constants import default_pg_timeout

from torchtune.utils.device import _get_device_from_env

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


def init_from_env(
    *,
    device_type: Optional[str] = None,
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
        device_type (Optional[str], optional): Device type to initialize. If None, device will be initialized
                                  based on environment. Supported device_types: "cpu", "cuda", "cuda:0", "cuda:1", etc.
        pg_backend (Optional[str], optional): The process group backend to use. If None, it will use the
                                    default process group backend from the device
        pg_timeout (timedelta, optional): Timeout for operations executed against the process
                                          group. Default value equals 30 minutes

    Returns:
        The current device.

    Raises:
        RuntimeError: If CUDA device type is specified, but CUDA is not available.
        RuntimeError: If torch.distributed is in use, but an indexed device is specified.
        RuntimeError: If the device type is specified but does not match the device type from the environment.
    """
    # Note: This will break when we need to support devices other than {CPU, CUDA}.
    if device_type is None or device_type == "cuda":
        device = _get_device_from_env()
    elif device_type == "cpu":
        device = torch.device("cpu")
    elif "cuda:" in device_type:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA is not available, but device specified is {device_type}"
            )
        else:
            device_type, device_index = device_type.split(":")
            device = torch.device(type=device_type, index=int(device_index))
            torch.cuda.set_device(device)

    if device_type is not None and device.type != device_type:
        raise RuntimeError(
            f"Device type is specified to {device_type} but got {device.type} from env"
        )

    if _check_dist_env():
        if not torch.distributed.is_available():
            _log.warning(
                "torch.distributed is not available. Skipping initializing the process group."
            )
            return device
        if torch.distributed.is_initialized():
            _log.warning(
                "torch.distributed is already initialized. Skipping initializing the process group."
            )
            return device
        pg_backend = (
            pg_backend
            if pg_backend is not None
            else _get_process_group_backend_from_device(device)
        )
        torch.distributed.init_process_group(backend=pg_backend, timeout=pg_timeout)
    return device


def _get_process_group_backend_from_device(device: torch.device) -> str:
    """Function that gets the default process group backend from the device."""
    return "nccl" if device.type == "cuda" else "gloo"


def get_world_size_and_rank() -> Tuple[int, int]:
    """Function that gets the current world size (aka total number
    of ranks) and rank number of the current trainer.

    Returns:
        Tuple[int, int]: world size, rank
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1, 0
    return torch.distributed.world_size(), torch.distributed.get_rank()


def seed(seed: int, debug_mode: Optional[Union[str, int]] = None) -> None:
    """Function that sets seed for pseudo-random number generators across commonly used libraries.

    This seeds PyTorch, NumPy, and the python.random module.
    For more details, see https://pytorch.org/docs/stable/notes/randomness.html.

    Args:
        seed (int): the integer value seed.
        debug_mode (Optional[Union[str, int]]): Controls debug_mode settings for deterministic operations within PyTorch.
            If `None`, don't set any PyTorch global values.
            If "default" or 0, don't error or warn on nondeterministic operations and additionally enable PyTorch CuDNN benchmark.
            If "warn" or 1, warn on nondeterministic operations and disable PyTorch CuDNN benchmark.
            If "error" or 2, error on nondeterministic operations and disable PyTorch CuDNN benchmark.
            For more details, see :func:`torch.set_deterministic_debug_mode` and
            https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms.

    Raises:
        ValueError: If the input seed value is outside the required range.
    """
    max_val = np.iinfo(np.uint32).max
    min_val = np.iinfo(np.uint32).min
    if seed < min_val or seed > max_val:
        raise ValueError(
            f"Invalid seed value provided: {seed}. Value must be in the range [{min_val}, {max_val}]"
        )
    _log.debug(f"Setting seed to {seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if debug_mode is not None:
        _log.debug(f"Setting deterministic debug mode to {debug_mode}")
        torch.set_deterministic_debug_mode(debug_mode)
        deterministic_debug_mode = torch.get_deterministic_debug_mode()
        if deterministic_debug_mode == 0:
            _log.debug("Disabling cuDNN deterministic mode")
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            _log.debug("Enabling cuDNN deterministic mode")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # reference: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
