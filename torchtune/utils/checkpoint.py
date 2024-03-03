# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torchtune.utils.constants import (
    CheckpointFormat,
    EPOCHS_KEY,
    MAX_STEPS_KEY,
    MODEL_KEY,
    OPT_KEY,
    SEED_KEY,
    TOTAL_EPOCHS_KEY,
)
from torchtune.utils.distributed import get_world_size_and_rank

# logger = torchtune.utils.get_logger("DEBUG")


def _contains_fsdp(model: nn.Module) -> bool:
    """
    Checks if the model contains FSDP.

    Args:
        model (nn.Module): Model to check.

    Returns:
        bool: True if the model contains FSDP, False otherwise.
    """
    return any(
        isinstance(m, torch.distributed.fsdp.FullyShardedDataParallel)
        for m in model.modules()
    )


def load_checkpoint(
    ckpt_path: Path,
    ckpt_format: CheckpointFormat,
    resume_from_checkpoint: bool,
) -> Dict[str, Any]:
    """ """
    if not ckpt_path.exists():
        raise ValueError(
            f"Checkpoint path {ckpt_path} does not exist. "
            "Please provide a valid checkpoint path."
        )
    if resume_from_checkpoint:
        ckpt_dict = _load_torchtune_checkpoint(ckpt_path, ckpt_format)
    else:
        ckpt_dict = _load_external_checkpoint(ckpt_path, ckpt_format)
    return ckpt_dict


def _load_torchtune_checkpoint(
    ckpt_path: Path,
    ckpt_format: CheckpointFormat,
) -> Dict[str, Any]:
    if ckpt_format != CheckpointFormat.TORCHTUNE_FORMAT:
        raise ValueError(
            "When resuming a TorchTune training run, the checkpoint format is expected to "
            f'be "torchtune". Got {ckpt_format} instead.'
        )
    if not ckpt_path.is_file():
        raise ValueError(
            "When resuming a TorchTune training run, the checkpoint path should "
            f"should point to a file. Got {ckpt_path} which is not a file."
        )
    if ckpt_path.suffix != ".pt":
        raise ValueError(
            'When resuming a TorchTune training run, the ckpt should be ".pt" file. '
            f'Got a "{ckpt_path.suffix}" file instead. Make sure you\'re loading a valid '
            "TorchTune checkpoint."
        )
    ckpt_dict = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
    return ckpt_dict


def _load_external_checkpoint(ckpt_path, ckpt_format):
    if ckpt_format == CheckpointFormat.META_FORMAT:
        state_dict = _fetch_meta_format_state_dict(ckpt_path)
    elif ckpt_format == CheckpointFormat.HF_FORMAT:
        state_dict = _fetch_hf_format_state_dict(ckpt_path)
    else:
        raise NotImplementedError(f"Checkpoint format {ckpt_format} not supported")
    return state_dict


def _fetch_meta_format_state_dict(ckpt_path):
    ckpt_files = []
    if ckpt_path.is_dir():
        ckpt_files = list(ckpt_path.glob("*.pth"))
        if len(ckpt_files) == 0:
            raise ValueError(
                "For meta format checkpoint, the directory should contain at least one .pth file. "
                "None found."
            )
    elif ckpt_path.is_file():
        if ckpt_path.suffix != ".pth":
            raise ValueError(
                'For meta format checkpoint, the file should be a ".pth" file. '
                f'Got a "{ckpt_path.suffix}" file instead.'
            )
        ckpt_files = [ckpt_path]
    else:
        raise ValueError("Unsupported value for checkpoint path")

    merged_state_dict = {}
    for ckpt_file in ckpt_files:
        state_dict = torch.load(
            ckpt_file, map_location="cpu", mmap=True, weights_only=True
        )
        merged_state_dict.update(state_dict)
    return merged_state_dict


def _fetch_hf_format_state_dict(ckpt_path):
    ckpt_files = []
    if ckpt_path.is_dir():
        ckpt_files = list(ckpt_path.glob("*.bin"))
        if len(ckpt_files) == 0:
            raise ValueError(
                "For meta format checkpoint, the directory should contain at least one .bin file. "
                "None found."
            )
    else:
        raise ValueError("Unsupported value for checkpoint path")

    merged_state_dict = {}
    for ckpt_file in ckpt_files:
        state_dict = torch.load(
            ckpt_file, map_location="cpu", mmap=True, weights_only=True
        )
        merged_state_dict.update(state_dict)
    return merged_state_dict


def save_checkpoint(
    ckpt_dict: Dict[str, Any],
    output_loc: str,
    model_key_filter: Optional[Callable[[str], bool]] = None,
) -> None:
    """
    Saves `ckpt_dict` to `output_loc`. `ckpt_dict` is expected to have at least a key `model` which represents
    the model to be checkpointed. This function will call `state_dict` in a distributed-aware fashion on checkpointable objects
    (currently only objects specified by "model" and "optimizer" keys). For distributed jobs, only rank 0
    will write out a checkpoint.
    Only full (unsharded) checkpoints are supported currently, i.e. full checkpoints are taken even if model and optimizer
    are sharded with FSDP.

    Args:
        ckpt_dict (Dict[str, Any]): Dictionary containing the checkpoint to be saved. Must have at least `model` key.
        output_loc (str): Local path to save the checkpoint to.
        model_key_filter (Optional[Callable[[str], bool]]): Optional function to filter the keys in the model state dict.
            This function should return True if the key is intended to be included in the saved checkpoint
            and False otherwise.
    Raises:
        RuntimeError: If `ckpt_dict` does not contain a `model` key.

    Example:
        >>> output_loc = "/tmp/output.pt"
        >>> ckpt_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        >>> torchtune.utils.checkpoint.save_checkpoint(ckpt_dict, output_loc)
    """
    if MODEL_KEY not in ckpt_dict:
        raise RuntimeError(
            "Expected `ckpt_dict` to contain a `model` key, but it does not."
        )
    if OPT_KEY in ckpt_dict:
        optimizer_state_dict = (
            FSDP.optim_state_dict(
                ckpt_dict[MODEL_KEY],
                ckpt_dict[OPT_KEY],
            )
            if _contains_fsdp(ckpt_dict[MODEL_KEY])
            else ckpt_dict[OPT_KEY].state_dict()
        )
        ckpt_dict[OPT_KEY] = optimizer_state_dict

    model_state_dict = ckpt_dict[MODEL_KEY].state_dict()
    if model_key_filter:
        model_state_dict = {
            k: v for k, v in model_state_dict.items() if model_key_filter(k)
        }
    ckpt_dict[MODEL_KEY] = model_state_dict
    _, rank = get_world_size_and_rank()
    if rank == 0:
        torch.save(ckpt_dict, output_loc)


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
        if _contains_fsdp(model)
        else opt_state_dict
    )

    return optim_state_dict_to_load


def validate_checkpoint(ckpt_dict: Dict[str, Any], resume_from_checkpoint: bool):
    """
    Validates the checkpoint dict. This includes validating the recipe state in case we're resuming
    training from a checkpoint.

    Args:
        ckpt_dict (Dict[str, Any]): Dictionary with recipe state, extracted from the checkpoint
        resume_from_checkpoint (bool): Boolean flag specifying whether training is being resumed from a checkpoint.

    Raises:
        RuntimeError: If ``ckpt_dict`` does not contain a ``model`` key.
        RuntimeError: If ``resume_from_checkpoint`` is `True` and `ckpt_dict` does not contain
            either "optimizer", "epochs_run", "seed", "total_epochs" or "max_steps_per_epoch" keys.
    """
    if MODEL_KEY not in ckpt_dict:
        raise RuntimeError(
            """Expected loaded checkpoint to contain a `model` key, but it does not. Ensure checkpoint was saved
            with `save_checkpoint`."""
        )

    if resume_from_checkpoint:

        # If the correct state is not available, fail. Training will not be
        # meaningful
        if (
            OPT_KEY not in ckpt_dict
            or EPOCHS_KEY not in ckpt_dict
            or SEED_KEY not in ckpt_dict
            or TOTAL_EPOCHS_KEY not in ckpt_dict
            or MAX_STEPS_KEY not in ckpt_dict
        ):
            raise ValueError(
                f"Checkpoint does not contain the required keys needed to resume training correctly.\n"
                f"Expected Keys: {OPT_KEY}, {EPOCHS_KEY}, {SEED_KEY}, {TOTAL_EPOCHS_KEY}, {MAX_STEPS_KEY}, {MODEL_KEY}\n"
                f"Found Keys: {ckpt_dict.keys()}."
            )
