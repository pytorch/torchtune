# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from safetensors import safe_open
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torchtune.utils._distributed import contains_fsdp


class ModelType(Enum):
    LLAMA2 = "llama2"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    LLAMA3 = "llama3"


def get_path(input_dir: Path, filename: str, missing_ok: bool = False) -> Path:
    """
    Utility to recover and validate the path for a given file within a given directory.

    Args:
        input_dir (Path): Directory containing the file
        filename (str): Name of the file
        missing_ok (bool): Whether to raise an error if the file is missing.

    Returns:
        Path: Path to the file

    Raises:
        ValueError: If the file is missing and missing_ok is False.
    """
    if not input_dir.is_dir():
        raise ValueError(f"{input_dir} is not a valid directory.")

    file_path = Path.joinpath(input_dir, filename)

    # If missing_ok is False, raise an error if the path is invalid
    if not missing_ok and not file_path.is_file():
        raise ValueError(f"No file with name: {filename} found in {input_dir}.")
    return file_path


def safe_torch_load(checkpoint_path: Path, weights_only: bool = True) -> Dict[str, Any]:
    """
    Utility to load a checkpoint file in a safe manner.
    """
    try:
        # convert the path into a string since pathlib Path and mmap don't work
        # well together
        is_safetensors_file = (
            True if str(checkpoint_path).endswith(".safetensors") else False
        )
        if is_safetensors_file:
            result = {}
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    result[k] = f.get_tensor(k)
            state_dict = result
        else:
            state_dict = torch.load(
                str(checkpoint_path),
                map_location="cpu",
                mmap=True,
                weights_only=weights_only,
            )
    except Exception as e:
        raise ValueError(f"Unable to load checkpoint from {checkpoint_path}. ") from e
    return state_dict


def transform_opt_state_dict(
    opt_state_dict: Dict[str, Any], model: nn.Module, optimizer: optim.Optimizer
) -> Dict[str, Any]:
    """
    Transforms the optimizer state dict for FSDP using the ``optim_state_dict_to_load``
    from distributed library within PyTorch. If FSDP is not used, the optimizer state dict is returned as is.

    Args:
        opt_state_dict (Dict[str, Any]): Optimizer state dict extracted from the checkpoint
        model (nn.Module): Model that checkpoint will be loaded into.
        optimizer (optim.Optimizer): Optimizer that optimizer state checkpoints will be loaded into.

    Returns:
        ckpt_dict (Dict[str, Any]): Transformed optimizer state dict.
    """
    optim_state_dict_to_load = (
        FSDP.optim_state_dict_to_load(model, optimizer, opt_state_dict)
        if contains_fsdp(model)
        else opt_state_dict
    )

    return optim_state_dict_to_load


def save_config(path: Path, config: Dict[str, Any]) -> None:
    """
    Save a configuration dictionary to a file.

    Args:
        path (Path): Path to save the configuration file.
        config (Dict[str, Any]): Configuration dictionary to save.
    """
    if not path.is_dir():
        path.mkdir(exist_ok=True)
    file_path = Path.joinpath(path, "config.json")
    if not file_path.exists():
        with open(file_path, "w") as f:
            json.dump(config, f)
