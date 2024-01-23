# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torchtune.utils.distributed import get_world_size_and_rank


def _contains_fsdp(model: torch.nn.Module) -> bool:
    """
    Checks if the model contains FSDP.

    Args:
        model (torch.nn.Module): Model to check.

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
    (currently only objects specified by "model" and "optimizer" keys). For distributed jobs, only rank 0 will write out a checkpoint.
    Only full checkpoints are supported currently, i.e. full checkpoints are taken even if model and optimizer are sharded with FSDP.

    Args:
        ckpt_dict (Dict[str, Any]): Dictionary containing the checkpoint to be saved. Must have at least `model` key.
        output_loc (str): Path to save the checkpoint to.
    """
    if "model" not in ckpt_dict:
        raise RuntimeError(
            "Expected `ckpt_dict` to contain a `model` key, but it does not."
        )
    model_state_dict = ckpt_dict["model"].state_dict()
    if "optimizer" in ckpt_dict:
        optimizer_state_dict = (
            FSDP.optim_state_dict(ckpt_dict["model"], ckpt_dict["optimizer"])
            if _contains_fsdp(ckpt_dict["model"])
            else ckpt_dict["optimizer"].state_dict()
        )
        ckpt_dict["optimizer"] = optimizer_state_dict

    ckpt_dict["model"] = model_state_dict
    _, rank = get_world_size_and_rank()
    if rank == 0:
        torch.save(ckpt_dict, output_loc)


def load_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """
    Loads a checkpoint from `ckpt_path` into `model` and optionally `optimizer`. This function is meant to be used in tandem with
    `save_checkpoint` and assumes the checkpoint was saved as such.

    Args:
        ckpt_path (str): String indicating path to saved checkpoint file. The checkpoint file is expected
        to have been saved with `save_checkpoint`.
        model (torch.nn.Module): Model to load the checkpoint into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load the checkpoint into. If not specified,
            "optimizer" key in `ckpt_dict` will be ignored, if present. Default: `None`.

    Returns:
        ckpt_dict (Dict[str, Any]): Dictionary containing loaded objects. Objects in this dictionary can be used
            to further restore non model and optimizer states.
    """

    ckpt_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" not in ckpt_dict:
        raise RuntimeError(
            "Expected loaded checkpoint to contain a `model` key, but it does not. Ensure checkpoint was saved with `save_checkpoint`."
        )
    if optimizer is not None and "optimizer" not in ckpt_dict:
        raise RuntimeError(
            "Expected loaded checkpoint to contain an `optimizer` key since an optimizer was passed in, but it does not. Ensure checkpoint was saved with `save_checkpoint`."
        )

    model.load_state_dict(ckpt_dict["model"])

    if optimizer is not None:
        optim_state_dict_to_load = (
            FSDP.optim_state_dict_to_load(model, optimizer, ckpt_dict["optimizer"])
            if _contains_fsdp(model)
            else ckpt_dict["optimizer"]
        )

        optimizer.load_state_dict(optim_state_dict_to_load)

    return ckpt_dict
