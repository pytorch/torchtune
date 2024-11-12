# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import random
from typing import Optional, Union

import numpy as np
import torch

from torchtune.training._distributed import _broadcast_tensor, get_world_size_and_rank
from torchtune.utils import get_logger

_log: logging.Logger = get_logger()


def set_seed(
    seed: Optional[int] = None, debug_mode: Optional[Union[str, int]] = None
) -> int:
    """Function that sets seed for pseudo-random number generators across commonly used libraries.

    This seeds PyTorch, NumPy, and the python.random module. For distributed jobs, each local process
    sets its own seed, computed seed + rank.
    For more details, see https://pytorch.org/docs/stable/notes/randomness.html.

    Args:
        seed (Optional[int]): the integer value seed. If `None`, a random seed will be generated and set.
        debug_mode (Optional[Union[str, int]]): Controls debug_mode settings for deterministic operations within PyTorch.

            * If `None`, don't set any PyTorch global values.
            * If "default" or 0, don't error or warn on nondeterministic operations and additionally enable PyTorch CuDNN benchmark.
            * If "warn" or 1, warn on nondeterministic operations and disable PyTorch CuDNN benchmark.
            * If "error" or 2, error on nondeterministic operations and disable PyTorch CuDNN benchmark.
            * For more details, see :func:`torch.set_deterministic_debug_mode` and
              https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms.

    Returns:
        int: the current seed

    Raises:
        ValueError: If the input seed value is outside the required range.
    """
    world_size, rank = get_world_size_and_rank()
    max_val = np.iinfo(np.uint32).max - world_size + 1
    min_val = np.iinfo(np.uint32).min
    if seed is None:
        rand_seed = torch.randint(min_val, max_val, (1,))
        seed = _broadcast_tensor(rand_seed, 0).item()  # sync seed across ranks
    if seed < min_val or seed > max_val:
        raise ValueError(
            f"Invalid seed value provided: {seed}. Value must be in the range [{min_val}, {max_val}]"
        )
    local_seed = seed + rank
    if rank == 0:
        _log.debug(
            f"Setting manual seed to local seed {local_seed}. Local seed is seed + rank = {seed} + {rank}"
        )

    torch.manual_seed(local_seed)
    np.random.seed(local_seed)
    random.seed(local_seed)

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

    return seed
