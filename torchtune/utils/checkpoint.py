# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torchtune.utils.constants import (
    EPOCHS_KEY,
    MAX_STEPS_KEY,
    MODEL_KEY,
    OPT_KEY,
    SEED_KEY,
    TOTAL_EPOCHS_KEY,
)
from torchtune.utils.distributed import get_world_size_and_rank


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


def save_checkpoint(ckpt_dict: Dict[str, Any], output_loc: str) -> None:
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
    ckpt_dict[MODEL_KEY] = model_state_dict
    _, rank = get_world_size_and_rank()
    if rank == 0:
        torch.save(ckpt_dict, output_loc)


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
) -> Dict[str, Any]:
    """
    Loads a checkpoint from `ckpt_path` into `model` and optionally `optimizer`. This function is meant to be used in tandem with
    `save_checkpoint` and assumes the checkpoint was saved as such. At minimum, the checkpoint needs to contain a `model` key that
    maps to the model's states.

    NOTE: `load_checkpoint` does NOT load model and optimizer states into the model and optimizer respectively.
    `load_checkpoint` handles the appropriate transformations (i.e. related to FSDP), but user is expected to
    call `load_state_dict` on the returned results.

    Args:
        ckpt_path (str): String indicating local path to saved checkpoint file.
        model (nn.Module): Model that checkpoint will be loaded into.
        optimizer (Optional[optim.Optimizer]): Optimizer that optimizer state checkpoints will be loaded into. If not specified,
            "optimizer" key in `ckpt_dict` will be ignored, if present. Default: `None`.

    Returns:
        ckpt_dict (Dict[str, Any]): Dictionary containing loaded objects. Objects in this dictionary can be used
            to restore model, optimizer, and any other checkpointed states.

    Raises:
        RuntimeError: If `ckpt_dict` does not contain a `model` key.
        RuntimeError: If `ckpt_dict` does not contain an `optimizer` key and an optimizer was passed in.

    Example:
        >>> ckpt_dict = torchtune.utils.checkpoint.load_checkpoint(ckpt_path, model, optimizer)
        >>> model.load_state_dict(ckpt_dict["model"])
        >>> optimizer.load_state_dict(ckpt_dict["optimizer"])
    """

    ckpt_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if MODEL_KEY not in ckpt_dict:
        raise RuntimeError(
            """Expected loaded checkpoint to contain a `model` key, but it does not. Ensure checkpoint was saved
            with `save_checkpoint`."""
        )
    if optimizer is not None and OPT_KEY not in ckpt_dict:
        raise RuntimeError(
            """Expected loaded checkpoint to contain an `optimizer` key since an optimizer was passed in, but it does not.
            Ensure checkpoint was saved with `save_checkpoint`."""
        )

    # Transform optimizer states if using FSDP and overwrite ckpt_dict["optimizer"] with the transformed optimizer state.
    if optimizer is not None:
        optim_state_dict_to_load = (
            FSDP.optim_state_dict_to_load(model, optimizer, ckpt_dict[OPT_KEY])
            if _contains_fsdp(model)
            else ckpt_dict[OPT_KEY]
        )

        ckpt_dict[OPT_KEY] = optim_state_dict_to_load

    return ckpt_dict


def transform_opt_state_dict(
    opt_state_dict: Dict[str, Any], model: nn.Module, optimizer: optim.Optimizer
) -> Dict[str, Any]:
    """
    TODO: Add docstring.
    """
    optim_state_dict_to_load = (
        FSDP.optim_state_dict_to_load(model, optimizer, opt_state_dict)
        if _contains_fsdp(model)
        else opt_state_dict
    )

    return optim_state_dict_to_load


def validate_checkpoint(ckpt_dict: Dict[str, Any], resume_from_checkpoint: bool):
    """
    Validates the checkpoint dict.
    """
    if MODEL_KEY not in ckpt_dict:
        raise RuntimeError(
            """Expected loaded checkpoint to contain a `model` key, but it does not. Ensure checkpoint was saved
            with `save_checkpoint`."""
        )

    if resume_from_checkpoint:
        # If optimizer state is not checkpointed, fail. Training will not be meaningful
        if OPT_KEY not in ckpt_dict:
            raise RuntimeError(
                """Expected loaded checkpoint to contain an `optimizer` key since training is being resumed,
                but it does not. Ensure checkpoint was saved with `save_checkpoint`."""
            )

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
