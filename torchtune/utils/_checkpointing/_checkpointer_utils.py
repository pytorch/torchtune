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
