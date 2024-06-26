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
from safetensors import safe_open


class ModelType(Enum):
    """ModelType is used by the checkpointer to distinguish between different model architectures.

    If you are adding a new model that follows a different format than those in the repo already,
    you can add a new ModelType to gate on weight conversion logic unique to that model.

    Example:
        >>> # Usage in a checkpointer class
        >>> def load_checkpoint(self, ...):
        >>>     ...
        >>>     if self._model_type == MY_NEW_MODEL:
        >>>         state_dict = my_custom_state_dict_mapping(state_dict)
    """

    GEMMA = "gemma"
    """Gemma family of models. See :func:`~torchtune.models.gemma.gemma`"""

    LLAMA2 = "llama2"
    """Llama2 family of models. See :func:`~torchtune.models.llama2.llama2`"""

    LLAMA3 = "llama3"
    """Llama3 family of models. See :func:`~torchtune.models.llama3.llama3`"""

    MISTRAL = "mistral"
    """Mistral family of models. See :func:`~torchtune.models.mistral.mistral`"""

    PHI3_MINI = "phi3_mini"
    """Phi-3 family of models. See :func:`~torchtune.models.phi3.phi3`"""

    MISTRAL_REWARD = "mistral_reward"
    """Mistral model with a classification head. See :func:`~torchtune.models.mistral.mistral_classifier`"""


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


def safe_torch_load(
    checkpoint_path: Path, weights_only: bool = True, mmap: bool = True
) -> Dict[str, Any]:
    """
    Utility to load a checkpoint file onto CPU in a safe manner. Provides separate handling for
    safetensors files.

    Args:
        checkpoint_path (Path): Path to the checkpoint file.
        weights_only (bool): Whether to load only tensors, primitive types, and dictionaries
            (passthrough to torch.load). Default: True
        mmap (bool): Whether to mmap from disk into CPU memory. Default: True

    Returns:
        Dict[str, Any]: State dict from the checkpoint file.

    Raises:
        ValueError: If the checkpoint file is not found or cannot be loaded.
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
                mmap=mmap,
                weights_only=weights_only,
            )
    except Exception as e:
        raise ValueError(f"Unable to load checkpoint from {checkpoint_path}. ") from e
    return state_dict


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
