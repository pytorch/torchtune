# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import string
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
from warnings import warn

import torch
from safetensors import safe_open

"""
Keys used during checkpoint load and checkpoint save.
"""

# adapter config containing info about LoRA modules, rank, alpha
ADAPTER_CONFIG = "adapter_config"
# key used for adapter weights such as LoRA weights
ADAPTER_KEY = "adapter"
# number of epochs completed thus far
EPOCHS_KEY = "epochs_run"
MAX_STEPS_KEY = "max_steps_per_epoch"
MODEL_KEY = "model"
OPT_KEY = "optimizer"
SEED_KEY = "seed"
# total number of epochs for training; resumed training runs for
# (total_epochs - epochs_run) number of epochs
TOTAL_EPOCHS_KEY = "total_epochs"
# number of steps completed thus far - for PPO
STEPS_KEY = "steps_run"
# rng state for ensuring correct training resuming in PPO
RNG_KEY = "rng_state"


class ModelType(Enum):
    """ModelType is used by the checkpointer to distinguish between different model architectures.

    If you are adding a new model that follows a different format than those in the repo already,
    you can add a new ModelType to gate on weight conversion logic unique to that model.

    Attributes:
        GEMMA (str): Gemma family of models. See :func:`~torchtune.models.gemma.gemma`
        LLAMA2 (str): Llama2 family of models. See :func:`~torchtune.models.llama2.llama2`
        LLAMA3 (str): Llama3 family of models. See :func:`~torchtune.models.llama3.llama3`
        LLAMA3_2 (str): Llama3.2 family of models. See :func:`~torchtune.models.llama3_2.llama3_2`
        LLAMA3_VISION (str): LLama3 vision family of models. See :func:`~torchtune.models.llama3_2_vision.llama3_2_vision_decoder`
        MISTRAL (str): Mistral family of models. See :func:`~torchtune.models.mistral.mistral`
        PHI3_MINI (str): Phi-3 family of models. See :func:`~torchtune.models.phi3.phi3`
        REWARD (str): A Llama2, Llama3, or Mistral model with a classification head projecting
            to a single class for reward modelling.
            See :func:`~torchtune.models.mistral.mistral_reward_7b` or :func:`~torchtune.models.llama2.llama2_reward_7b`
        QWEN2 (str): Qwen2 family of models. See :func:`~torchtune.models.qwen2.qwen2`

    Example:
        >>> # Usage in a checkpointer class
        >>> def load_checkpoint(self, ...):
        >>>     ...
        >>>     if self._model_type == MY_NEW_MODEL:
        >>>         state_dict = my_custom_state_dict_mapping(state_dict)
    """

    GEMMA: str = "gemma"
    LLAMA2: str = "llama2"
    LLAMA3: str = "llama3"
    LLAMA3_2: str = "llama3_2"
    LLAMA3_VISION: str = "llama3_vision"
    MISTRAL: str = "mistral"
    PHI3_MINI: str = "phi3_mini"
    REWARD: str = "reward"
    QWEN2: str = "qwen2"


class FormattedCheckpointFiles:
    """
    This class gives a more concise way to represent a list of filenames of the format ``file_{i}_of_{n_files}.pth``.

    Args:
        filename_format (str): Format string for the filename. Must have exactly two placeholders, e.g.
            ``file_{}_of_{}.pth``.
        max_filename (str): Maximum filename in the list. Should be a string representation of an integer,
            possibly with leading zeroes.
    """

    def __init__(
        self,
        filename_format: str,
        max_filename: str,
    ):
        self.filename_format = filename_format
        self.max_filename = max_filename
        self._validate_filename_format()

    @classmethod
    def from_dict(cls, d: dict) -> "FormattedCheckpointFiles":
        if "filename_format" not in d or "max_filename" not in d:
            raise ValueError(
                "Must pass 'filename_format' and 'max_filename' keys to generate checkpoint filenames"
            )
        return cls(
            filename_format=d["filename_format"],
            max_filename=d["max_filename"],
        )

    def _validate_filename_format(self):
        n_format_placeholders = [
            x[1]
            for x in string.Formatter().parse(self.filename_format)
            if x[1] is not None
        ]
        if len(n_format_placeholders) != 2:
            raise ValueError(
                "Filename format string must have exactly two placeholders, e.g. 'file_{i}_of_{n_files}.pth'"
            )

    def build_checkpoint_filenames(self):
        """
        Builds a list of checkpoint filenames from the filename format and max filename.

        Returns:
            List[str]: List of checkpoint filenames.

        Example:
            >>> # Example usage
            >>> f = FormattedCheckpointFiles(filename_format="file_{}_of_{}.safetensors", max_filename="00003")
            >>> f.build_checkpoint_filenames()
            >>> ['file_00001_of_00003.safetensors', 'file_00002_of_00003.safetensors', 'file_00003_of_00003.safetensors']
        """
        num_files = int(self.max_filename)
        return [
            self.filename_format.format(
                str(i + 1).zfill(len(self.max_filename)),
                self.max_filename,
            )
            for i in range(num_files)
        ]


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


def update_state_dict_for_classifier(
    state_dict: Dict[str, torch.Tensor],
    model_named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    force_override: bool = False,
):
    """
    Validates the state dict for checkpoint loading for a classifier model.
    To be used prior to a call to ``model.load_state_dict(state_dict)``.
    This function will overwrite the ``output.weight`` in the state-dict
    to be loaded with the ``output.weight`` in the model if the shapes
    for the ``output.weight`` do not match. You may also wish to override this behaviour,
    for example, if ``num_classes`` for your checkpoint and model are the same.

    Concretely, when fine-tuning a classifier model from the checkpoint of a base language model
    which has ``output.weight`` of shape ``[vocab_dim, embed_dim]``, we overwrite
    the ``output.weight`` in the state-dict to be loaded with the randomly initialized
    ``[num_classes, embed_dim]`` weight in the model. This is done in-place.

    Args:
        state_dict (Dict[str, torch.Tensor]): state dict to be loaded into the classifier model.
        model_named_parameters (Iterable[Tuple[str, torch.nn.Parameter]]): model named parameters
            from ``model.named_parameters()``.
        force_override (bool): Whether to replace ``output.weight`` in ``state_dict`` with the model's
            ``output.weight``, even if the shapes match.
    Notes:
        - ``output.bias`` will be ignored if present in ``state_dict``
        - This function will always replace the ``output.weight`` in ``state_dict``,
            if ``output.weight != model.output.weight``.

    Raises:
        AssertionError: if ``state_dict`` does not contain ``output.weight``.
        AssertionError: if ``model_named_parameters`` does not contain ``output.weight``.

    """
    output_weight = dict(model_named_parameters).get("output.weight", None)
    if "output.weight" not in state_dict:
        raise AssertionError(
            "Expected output.weight in state_dict, but it wasn't found."
        )
    if output_weight is None:
        raise AssertionError(
            "Expected output.weight in model_named_parameters, but it wasn't found."
        )
    if "output.bias" in state_dict:
        warn("Found output.bias in state dict - this will not be used!")
        state_dict.pop("output.bias")
    if state_dict["output.weight"].shape[0] != output_weight.shape[0] or force_override:
        state_dict["output.weight"] = output_weight
