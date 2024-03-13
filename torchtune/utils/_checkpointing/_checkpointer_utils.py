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


def get_recipe_checkpoint_path(checkpoint_dir: Path, filename: str) -> Path:
    """
    Utility to recover and validate the path for the recipe_state checkpoint file.
    """
    if not checkpoint_dir.is_dir():
        raise ValueError(
            f"Checkpoint directory {checkpoint_dir} is not a valid directory."
        )

    recipe_state_checkpoint_path = Path.joinpath(checkpoint_dir, filename)

    if not recipe_state_checkpoint_path.is_file():
        raise ValueError(
            "To resume training from checkpoint a valid recipe state file is needed. "
            f"No file with name: {filename} found in {checkpoint_dir}."
        )
    return recipe_state_checkpoint_path


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
