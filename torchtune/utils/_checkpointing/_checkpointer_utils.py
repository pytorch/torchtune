# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from pathlib import Path
from typing import Any, Dict

import torch


class ModelType(Enum):
    LLAMA2 = "llama2"


class OptimizerInBackwardWrapper:
    """
    A bare-bones class meant for checkpoint save and load for optimizers running
    in backward. Usage is limited to the following:

    optim_dict = {
        p: config.instantiate(cfg_optimizer, [p])
        for p in self._model.parameters()
    }
    # Save checkpoint
    ckpt = OptimizerInBackwardWrapper(optim_dict).state_dict()
    torch.save("/tmp/optim_ckpt", ckpt)
    # Load checkpoint
    placeholder_optim_dict = {
        p: config.instantiate(cfg_optimizer, [p])
        for p in self._model.parameters()
    }
    wrapper = OptimInBackwardWrapper(placeholder_optim_dict)
    # load_state_dict expects a dict produced by this class's
    # state_dict method.
    wrapper.load_state_dict(torch.load("/tmp/optim_ckpt"))
    # placeholder_optim_dict now has updated optimizer states.

    NOTE: This wrapper is only meant to be used for single-device use cases.
        Distributed use cases such as FSDP, which require specialized
        optimizer state checkpointing, are not supported.

    """

    def __init__(self, optim_map: Dict[str, torch.optim.Optimizer]):
        self.optim_map = optim_map

    def state_dict(self):
        return {p: opt.state_dict() for p, opt in self.optim_map.items()}

    def load_state_dict(self, optim_ckpt_map: Dict[str, Any]):
        params_covered = set()
        for param_name in optim_ckpt_map.keys():
            if param_name not in self.optim_map:
                raise RuntimeError(
                    f"Trying to load optimizer state for unexpected param {param_name}"
                )
            self.optim_map[param_name].load_state_dict(optim_ckpt_map[param_name])
            params_covered.add(param_name)
        # Ensure all params have been loaded into, report missing params
        missing_params = set(self.optim_map.keys()) - params_covered
        if missing_params:
            raise RuntimeError(
                f"Expected to load optimizer state for params {missing_params}!"
            )


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


def safe_torch_load(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Utility to load a checkpoint file in a safe manner.
    """
    try:
        # convert the path into a string since pathlib Path and mmap don't work
        # well together
        state_dict = torch.load(
            str(checkpoint_path), map_location="cpu", mmap=True, weights_only=True
        )
    except Exception as e:
        raise ValueError(f"Unable to load checkpoint from {checkpoint_path}. ") from e
    return state_dict
