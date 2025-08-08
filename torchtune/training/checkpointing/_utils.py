# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import shutil
import string
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional, Union
from warnings import warn

import torch
from fsspec.core import url_to_fs
from huggingface_hub import HfFileSystem
from safetensors import safe_open

from torchtune.utils._logging import get_logger

logger = get_logger("DEBUG")

"""
Keys used during checkpoint load and checkpoint save.
"""

# adapter config containing info about LoRA modules, rank, alpha
ADAPTER_CONFIG = "adapter_config"

# default used by huggingface when looking for saved files
# https://github.com/huggingface/peft/blob/d13d7a401ccf4808aaaf76480fea09a4cf4ac1f5/src/peft/config.py#L259C21-L259C32
ADAPTER_CONFIG_FNAME = "adapter_config"
ADAPTER_MODEL_FNAME = "adapter_model"
SAFETENSOR_INDEX_FNAME = "model.safetensors.index.json"
TORCH_INDEX_FNAME = "pytorch_model.bin.index.json"

# standardize checkpointing
SHARD_FNAME = "model-{cpt_idx}-of-{num_shards}"
RECIPE_STATE_DIRNAME = "recipe_state"

# Needed when setting up output dir in checkpointing
REPO_ID_FNAME = "original_repo_id"
SUFFIXES_TO_NOT_COPY = [
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    SAFETENSOR_INDEX_FNAME,
    TORCH_INDEX_FNAME,
    ADAPTER_CONFIG_FNAME,
    ADAPTER_MODEL_FNAME,
]

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

# key used for dataloader state
DATALOADER_KEY = "dataloader"
VAL_DATALOADER_KEY = "val_dataloader"


class ModelType(Enum):
    """ModelType is used by the checkpointer to distinguish between different model architectures.

    If you are adding a new model that follows a different format than those in the repo already,
    you can add a new ModelType to gate on weight conversion logic unique to that model.

    Attributes:
        GEMMA (str): Gemma family of models. See :func:`~torchtune.models.gemma.gemma`
        GEMMA2 (str): Gemma 2 family of models. See :func:`~torchtune.models.gemma2.gemma2`
        GEMMA3 (str): Gemma 3 family of models. See :func:`~torchtune.models.gemma3.gemma3`
        LLAMA2 (str): Llama2 family of models. See :func:`~torchtune.models.llama2.llama2`
        LLAMA3 (str): Llama3 family of models. See :func:`~torchtune.models.llama3.llama3`
        LLAMA3_2 (str): Llama3.2 family of models. See :func:`~torchtune.models.llama3_2.llama3_2`
        LLAMA3_VISION (str): LLama3 vision family of models. See :func:`~torchtune.models.llama3_2_vision.llama3_2_vision_decoder`
        LLAMA4 (str): Llama4 family of models. See :func:`~torchtune.models.llama4.llama4`
        MISTRAL (str): Mistral family of models. See :func:`~torchtune.models.mistral.mistral`
        PHI3_MINI (str): Phi-3 family of models. See :func:`~torchtune.models.phi3.phi3`
        PHI4 (str): Phi-4 family of models. See :func:`~torchtune.models.phi4.phi4`
        REWARD (str): A Llama2, Llama3, or Mistral model with a classification head projecting
            to a single class for reward modelling.
            See :func:`~torchtune.models.mistral.mistral_reward_7b` or :func:`~torchtune.models.llama2.llama2_reward_7b`
        QWEN2 (str): Qwen2 family of models. See :func:`~torchtune.models.qwen2.qwen2`
        CLIP_TEXT (str): CLIP text encoder. See :func:`~torchtune.models.clip.clip_text_encoder_large`
        T5_ENCODER (str): T5 text encoder. See :func:`~torchtune.models.t5.t5_v1_1_xxl_encoder`
        QWEN3 (str): Qwen3 family of models. See :func:`~torchtune.models.qwen3.qwen3`

    Example:
        >>> # Usage in a checkpointer class
        >>> def load_checkpoint(self, ...):
        >>>     ...
        >>>     if self._model_type == MY_NEW_MODEL:
        >>>         state_dict = my_custom_state_dict_mapping(state_dict)
    """

    GEMMA: str = "gemma"
    GEMMA2: str = "gemma2"
    GEMMA3: str = "gemma3"
    LLAMA2: str = "llama2"
    LLAMA3: str = "llama3"
    LLAMA3_2: str = "llama3_2"
    LLAMA3_VISION: str = "llama3_vision"
    LLAMA4: str = "llama4"
    MISTRAL: str = "mistral"
    PHI3_MINI: str = "phi3_mini"
    PHI4: str = "phi4"
    REWARD: str = "reward"
    QWEN2: str = "qwen2"
    CLIP_TEXT: str = "clip_text"
    T5_ENCODER: str = "t5_encoder"
    QWEN3: str = "qwen3"


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
        if not isinstance(d["max_filename"], str):
            raise ValueError(
                f"`max_filename` must be a string, but found {type(d['max_filename'])} instead."
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
            list[str]: list of checkpoint filenames.

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


def get_path(
    input_dir: Union[Path, str], filename: str, missing_ok: bool = False
) -> str:
    """
    Utility to recover and validate the path for a given file within a given directory.

    Args:
        input_dir (Union[Path, str]): Directory containing the file
        filename (str): Name of the file
        missing_ok (bool): Whether to raise an error if the file is missing.

    Returns:
        str: Path to the file

    Raises:
        ValueError: If the file is missing and missing_ok is False.
    """
    fs, _ = url_to_fs(input_dir)
    if not fs.isdir(input_dir):
        raise ValueError(f"{input_dir} is not a valid directory.")

    file_path = os.path.join(input_dir, filename)

    # If missing_ok is False, raise an error if the path is invalid
    if not missing_ok and not fs.isfile(file_path):
        raise ValueError(f"No file with name: {filename} found in {input_dir}.")
    return file_path


def safe_torch_load(
    checkpoint_path: Union[Path, str], weights_only: bool = True, mmap: bool = True
) -> dict[str, Any]:
    """
    Utility to load a checkpoint file onto CPU in a safe manner. Provides separate handling for
    safetensors files.

    Args:
        checkpoint_path (Union[Path, str]): Path to the checkpoint file.
        weights_only (bool): Whether to load only tensors, primitive types, and dictionaries
            (passthrough to torch.load). Default: True
        mmap (bool): Whether to mmap from disk into CPU memory. Default: True

    Returns:
        dict[str, Any]: State dict from the checkpoint file.

    Raises:
        ValueError: If the checkpoint file is not found or cannot be loaded.
    """
    try:
        # convert the path into a string since pathlib Path and mmap don't work
        # well together
        fs, _ = url_to_fs(str(checkpoint_path))
        is_safetensors_file = (
            True if str(checkpoint_path).endswith(".safetensors") else False
        )
        if is_safetensors_file:
            state_dict = {}
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        else:
            if isinstance(fs, HfFileSystem):
                # HfFileSystem does not support mmap
                mmap = False
                with fs.open(checkpoint_path, "rb") as checkpoint_file:
                    state_dict = torch.load(
                        checkpoint_file,
                        map_location="cpu",
                        mmap=mmap,
                        weights_only=weights_only,
                    )
            else:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    mmap=mmap,
                    weights_only=weights_only,
                )
    except Exception as e:
        raise ValueError(f"Unable to load checkpoint from {checkpoint_path}. ") from e
    return state_dict


def update_state_dict_for_classifier(
    state_dict: dict[str, torch.Tensor],
    model_named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
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
        state_dict (dict[str, torch.Tensor]): state dict to be loaded into the classifier model.
        model_named_parameters (Iterable[tuple[str, torch.nn.Parameter]]): model named parameters
            from ``model.named_parameters()``.
        force_override (bool): Whether to replace ``output.weight`` in ``state_dict`` with the model's
            ``output.weight``, even if the shapes match.
    Notes:
        - ``output.bias`` will be ignored if present in ``state_dict``
        - This function will always replace the ``output.weight`` in ``state_dict``,
            if ``output.weight != model.output.weight``.

    Raises:
        AssertionError:
            If ``state_dict`` does not contain ``output.weight``, **or**
            if ``model_named_parameters`` does not contain ``output.weight``.

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


# TODO: add test
def get_largest_iter_folder(
    dir: Union[str, Path], pattern: str = r"^epoch_(\d+)"
) -> Union[None, str]:
    largest_iter_folder = None
    iter_folders = []
    regex = re.compile(pattern)

    fs, _ = url_to_fs(dir)
    # Iterate over the directory contents
    for fpath in fs.ls(dir, detail=False):
        fname = os.path.basename(fpath)
        match = regex.match(fname)
        if match:
            # Extract the number from the match
            iter_number = int(match.group(1))
            iter_folders.append((fname, iter_number))

    # Find the folder with the largest iter number
    if iter_folders:
        largest_iter_folder = max(iter_folders, key=lambda x: x[1])[0]

    return largest_iter_folder


# TODO: instead of copying, make it a symlink when we start using HF cache
def copy_files(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    ignore_suffixes: Optional[list[str]] = None,
    max_file_size_mb: int = 100,
) -> None:
    """
    Copies files from the input directory to the output directory, preserving the directory structure.

    This function will skip copying files that already exist in the output directory or have specific suffixes.
    It will also skip folders and files that start with '.'. E.g. ".cache/" and ".git".

    Args:
        input_dir (Union[str, Path]): The path to the input directory containing files to be copied.
        output_dir (Union[str, Path]): The path to the output directory where files should be copied.
        ignore_suffixes (Optional[list[str]]): A list of file suffixes to exclude from copying.
          Defaults to ['.pt', '.bin', '.safetensors'] if not provided.
        max_file_size_mb (int): The maximum file size in megabytes to copy. Defaults to 100 MB.
    Returns:
        None
    Example:
    >>> copy_files('path/to/input_dir', 'path/to/output_dir')

    This will copy all files from 'path/to/input_dir' to 'path/to/output_dir', except those that
    already exist in the destination or have the specified suffixes.
    """
    fs, _ = url_to_fs(input_dir)
    max_file_size = max_file_size_mb * 1024 * 1024
    for root, dirs, files in fs.walk(input_dir):
        # Filter out directories that start with '.'. E.g. ".cache/"
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        # Construct the corresponding directory in the output
        protocol = fs.protocol if isinstance(fs.protocol, tuple) else (fs.protocol)
        if "local" in protocol:
            relative_path = os.path.relpath(root, input_dir)
            dest_dir = os.path.join(output_dir, relative_path)
        else:
            dest_dir = output_dir

        # Create the directory in the output if it doesn't exist
        fs.makedirs(dest_dir, exist_ok=True)

        for file in files:
            # Skip files that start with '.'. E.g. ".git"
            if file.startswith("."):
                continue

            # Check if the file has one of the specified suffixes
            if ignore_suffixes and any(
                file.endswith(suffix) for suffix in ignore_suffixes
            ):
                continue

            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_dir, file)

            # Check the file size
            if fs.size(src_file) > max_file_size:
                print(
                    f"Skipping copying {src_file} to {output_dir} as it exceeds the size limit of {max_file_size_mb} MiB."
                )
                continue

            # Copy the file if it doesn't already exist in the destination
            if not fs.exists(dest_file):
                fs.cp_file(src_file, dest_file)

    return


def get_recipe_checkpoint_path(
    output_dir: Union[str, Path],
    checkpoint_dir: Union[str, Path],
    recipe_checkpoint: Optional[str] = None,
    should_load_recipe_state: bool = False,
) -> Optional[str]:
    """
    If recipe_checkpoint is None, look for recipe_state.pt in {checkpoint_dir/recipe_state.pt}.
    This is to make it easier to resume from a previous run, without having to specify the recipe_checkpoint.

    Args:
        output_dir (Union[str, Path]): Directory containing the directory that has the recipe checkpoint.
        checkpoint_dir (Union[str, Path]): Directory containing the recipe checkpoint.
        recipe_checkpoint (Optional[str]): Name of the recipe checkpoint file. Defaults to None.
        should_load_recipe_state (bool): Whether to load the recipe state from the checkpoint.
    Returns:
        Optional[str]: Path to the recipe checkpoint file if should_load_recipe_state is True, otherwise None.
    Raises:
        ValueError: If should_load_recipe_state is True and the recipe checkpoint file is missing.
    """
    if not should_load_recipe_state:
        return None

    recipe_checkpoint_path = None
    if recipe_checkpoint:
        recipe_checkpoint_path = os.path.join(output_dir, recipe_checkpoint)
    else:
        recipe_checkpoint_path = os.path.join(checkpoint_dir, "recipe_state.pt")

    fs, _ = url_to_fs(recipe_checkpoint_path)

    if not recipe_checkpoint_path or not fs.exists(recipe_checkpoint_path):
        raise ValueError(
            "If `should_load_recipe_state=True`, recipe_checkpoint file must be provided. "
            f"Could not find it at {recipe_checkpoint_path}."
        )

    return recipe_checkpoint_path


def get_adapter_checkpoint_path(
    output_dir: Union[Path, str],
    adapter_checkpoint: Optional[str] = None,
    should_load_recipe_state: bool = False,
    pattern: str = r"^epoch_(\d+)",
) -> Optional[str]:
    r"""
    If adapter_checkpoint is None, look for it in {output_dir}/epoch_{latest_epoch}/adapter_model.pt.
    This is to make it easier to resume from a previous run, without having to specify the adapter_checkpoint.

    Args:
        output_dir (Union[Path, str]): Directory containing the adapter checkpoint.
        adapter_checkpoint (Optional[str]): Name of the adapter checkpoint file. Defaults to None.
        should_load_recipe_state (bool): Whether to load the recipe state from checkpoint.
        pattern (str): Regex pattern to match the epoch folder. Defaults to "epoch_(\d+)".

    Returns:
        Optional[str]: Path to the adapter checkpoint file, or None if not applicable.

    Raises:
        ValueError: If the adapter checkpoint file is missing or if the adapter checkpoint file is not a .pt file.
    """
    if not should_load_recipe_state:
        return None

    adapter_checkpoint_path = None

    if adapter_checkpoint:
        adapter_checkpoint_path = os.path.join(output_dir, adapter_checkpoint)
        fs, _ = url_to_fs(adapter_checkpoint_path)
        if not fs.exists(adapter_checkpoint_path):
            raise ValueError(
                f"Adapter checkpoint file {adapter_checkpoint_path} does not exist."
            )
        if not adapter_checkpoint_path.endswith(".pt"):
            raise ValueError(
                f"Adapter checkpoint file {adapter_checkpoint_path} must end with .pt extension."
            )
    else:
        # Look for the latest adapter checkpoint in the output directory
        largest_iter_folder = get_largest_iter_folder(output_dir, pattern=pattern)
        if largest_iter_folder is None:
            return None

        tentative_adapter_checkpoint_path = os.path.join(
            output_dir, largest_iter_folder, "adapter_model.pt"
        )
        fs, _ = url_to_fs(tentative_adapter_checkpoint_path)
        if fs.exists(tentative_adapter_checkpoint_path):
            adapter_checkpoint_path = tentative_adapter_checkpoint_path

    return adapter_checkpoint_path if adapter_checkpoint_path else None


def get_model_checkpoint_path(
    checkpoint_files: Union[list[str], dict[str, str]],
    checkpoint_dir: Union[str, Path],
    output_dir: Union[str, Path],
    should_load_recipe_state: bool,
    has_adapter_checkpoint: bool,
) -> list[Path]:
    """
    Returns Paths to model checkpoint files, handling resuming from checkpoint, file formating and checking
    if the files exists.

    If resuming from checkpoint, the checkpoint files are loaded from the output directory. Otherwise,
    they are loaded from the checkpoint directory.

    If checkpoint_fiels is a dictionary, it is converted to a list of formatted checkpoint filenames.

    Args:
        checkpoint_files (Union[list[str], dict[str, str]]): list or dictionary of checkpoint file names.
            If a dictionary with keys ["filename_format", "max_filename"] is provided,
            it is converted to a list of formatted checkpoint filenames.
        checkpoint_dir (Union[str, Path]): Directory containing the checkpoint files.
        output_dir (Union[str, Path]): Directory to use when resuming from a checkpoint.
        should_load_recipe_state (bool): Whether to resume from a checkpoint.
        has_adapter_checkpoint (bool): Indicates if there is an adapter checkpoint.
    Returns:
        list[Path]: Sorted list of paths to the checkpoint files.
    Example:
        >>> checkpoint_files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        >>> checkpoint_dir = "/path/to/checkpoints"
        >>> output_dir = "/path/to/output"
        >>> should_load_recipe_state = True
        >>> has_adapter_checkpoint = False
        >>> paths = get_model_checkpoint_path(
        ...     checkpoint_files,
        ...     checkpoint_dir,
        ...     output_dir,
        ...     should_load_recipe_state,
        ...     has_adapter_checkpoint
        ... )
        >>> print(paths)
        [PosixPath('/path/to/output/{largest_epoch}/model-00001-of-00002.safetensors'),
         PosixPath('/path/to/output/{largest_epoch}/model-00002-of-00002.safetensors')]
    """

    def validate_checkpoint_files(
        checkpoint_files: Union[list[str]],
        input_dir: Optional[Path],
        missing_ok=False,
    ) -> list[Path]:
        """
        Validates that the checkpoint files exist and sorts based on ID.
        """

        checkpoint_paths: list[Path] = []
        for f in checkpoint_files:
            checkpoint_path = get_path(input_dir, f, missing_ok)
            checkpoint_paths.append(checkpoint_path)

        return sorted(checkpoint_paths)

    # load or resume from model weights

    # e.g.
    # checkpoint_files:
    #   filename_format: model-{}-of-{}.safetensors
    #   max_filename: "00191"
    # becomes checkpoint_files = [model-00001-of-00191.safetensors, model-00002-of-00191,..]
    if not isinstance(checkpoint_files, list):
        # TODO: this can be a function instead of a class
        formatted_checkpoint_files = FormattedCheckpointFiles.from_dict(
            checkpoint_files
        )
        checkpoint_files = formatted_checkpoint_files.build_checkpoint_filenames()

    # Case 1: not loading the recipe state
    if not should_load_recipe_state:
        input_dir = checkpoint_dir

    # Case 2: Loading the recipe state, but its full finetuning (no adapter)
    elif not has_adapter_checkpoint:
        input_dir = output_dir

    # Case 3: Loading the recipe state and has an adapter.
    else:
        # FIXME
        # TODO: if the model has lora + trained weights, e.g. embeddings,
        # we will silently not load the trained model, because we load from checkpoint_dir.
        # We cannot load from output_dir because we always merge the adapter weights into the model
        input_dir = checkpoint_dir

    checkpoint_paths = validate_checkpoint_files(
        checkpoint_files,
        input_dir=input_dir,
        missing_ok=False,
    )

    return checkpoint_paths


def check_outdir_not_in_ckptdir(
    ckpt_dir: Union[Path, str], out_dir: Union[Path, str]
) -> bool:
    """
    Checks that the output directory is not equal to or a subdirectory of the checkpoint directory.
    This is necessary to avoid making copies of copies when geting config files from ckpt_dir.
    """
    # Resolve the absolute paths to avoid issues with relative paths
    if isinstance(ckpt_dir, Path):
        _ckpt_dir = ckpt_dir.resolve()
    if isinstance(out_dir, Path):
        _out_dir = out_dir.resolve()

    _ckpt_dir = Path(ckpt_dir)
    _out_dir = Path(out_dir)

    # Check if out_dir is the same as ckpt_dir or a subdirectory of it
    if _out_dir == _ckpt_dir or _ckpt_dir in _out_dir.parents:
        raise ValueError(
            "The output directory cannot be the same as or a subdirectory of the checkpoint directory. "
            f"Found {ckpt_dir=} and {out_dir=}."
        )

    return True


def get_all_checkpoints_in_dir(
    dir: Path, *, pattern: str = r"^epoch_(\d+)"
) -> list[Path]:
    """
    Returns a list of all checkpoints in the given directory.
    The pattern argument is a regular expression that matches the epoch number in the checkpoint filename.
    The default pattern matches filenames of the form "epoch_{epoch_number}".

    Args:
        dir (Path): The directory containing the checkpoints.
        pattern (str): A regular expression pattern to match the epoch number in the checkpoint filename.
            Defaults to "epoch_(\\d+)".

    Example:
        >>> dir = Path("/path/to/checkpoints")
        >>> pattern = r"^epoch_(\\d+)"
        >>> get_all_checkpoints_in_dir(dir, pattern=pattern)
        [PosixPath('/path/to/checkpoints/epoch_1'), PosixPath('/path/to/checkpoints/epoch_2'), ...]

    Returns:
        list[Path]: A list of Path objects representing the checkpoints..
    """
    checkpoints = []
    regex_to_match = re.compile(pattern)

    # Iterate over the directory contents
    for item in dir.iterdir():
        if item.is_dir():
            # Check if the directory name matches the pattern
            match = regex_to_match.match(item.name)
            if match:
                checkpoints.append(item)

    return checkpoints


def prune_surplus_checkpoints(
    checkpoints: list[Path], keep_last_n_checkpoints: int = 1
) -> None:
    """
    Prunes the surplus checkpoints in the given list of checkpoints.
    The function will keep the latest checkpoints based on the param `keep_last_n_checkpoints` and delete the rest.

    Args:
        checkpoints (list[Path]): A list of Path objects representing the checkpoints.
        keep_last_n_checkpoints (int): The number of checkpoints to keep. Defaults to 1.

    Note:
        Expects the format of the checkpoints to be "epoch_{epoch_number}" or "step_{step_number}". A higher number
        indicates a more recent checkpoint. E.g. "epoch_1" is more recent than "epoch_0".

    Example:
        >>> checkpoints = [PosixPath('/path/to/checkpoints/epoch_1'), PosixPath('/path/to/checkpoints/epoch_2')]
        >>> prune_surplus_checkpoints(checkpoints, keep_last_n_checkpoints=1)
        >>> os.listdir('/path/to/checkpoints')
        ['epoch_2']

    Raises:
        ValueError: If `keep_last_n_checkpoints` is less than 1.
    """
    if keep_last_n_checkpoints < 1:
        raise ValueError("keep_last_n_checkpoints must be greater than or equal to 1.")

    # Sort the checkpoints by their epoch or step number
    checkpoints.sort(key=lambda x: int(x.name.split("_")[-1]), reverse=True)

    # Delete the surplus checkpoints
    for checkpoint in checkpoints[keep_last_n_checkpoints:]:
        shutil.rmtree(checkpoint)

    return


def get_most_recent_checkpoint(dir: Path) -> Optional[Path]:
    """
    Return the most recent checkpoint in the given directory.
    The function assumes that the checkpoint files are named in the format "epoch_{epoch_number}" or "step_{step_number}".
    The function will return None if no checkpoint files are found in the directory.

    Args:
        dir (Path): The directory containing the checkpoints.

    Returns:
        Optional[Path]: The path to the most recent checkpoint, or None if no checkpoints are found.
    """
    # First, check for epochs
    checkpoints = get_all_checkpoints_in_dir(dir, pattern=r"^epoch_(\d+)")

    # If no epochs found, check for steps
    if not checkpoints:
        checkpoints = get_all_checkpoints_in_dir(dir, pattern=r"^step_(\d+)")

    # If no steps found, return None
    if not checkpoints:
        return None

    # Finally, loop through checkpoints and return the most recent (non-empty) one
    checkpoints.sort(key=lambda x: int(x.name.split("_")[-1]))
    while checkpoints:
        ckpt = checkpoints.pop()
        if any(ckpt.iterdir()):
            return ckpt
