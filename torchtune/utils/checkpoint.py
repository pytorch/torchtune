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

from torchtune.utils.constants import MODEL_KEY, OPT_KEY
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
    model_state_dict = ckpt_dict[MODEL_KEY].state_dict()
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
            FSDP.optim_state_dict_to_load(
                model, optimizer, ckpt_dict[OPT_KEY]
            )
            if _contains_fsdp(model)
            else ckpt_dict[OPT_KEY]
        )

        ckpt_dict[OPT_KEY] = optim_state_dict_to_load

    return ckpt_dict


def load_checkpoint_updated(
    ckpt_path: str,
    resume_from_checkpoint: bool,
    model: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
) -> Dict[str, Any]:
    """
    Loads a checkpoint from `ckpt_path`.

    This function makes the following assumptions:
        - When `resume_from_checkpoint` is `False`, the pre-trained checkpoint being loaded is a "native" checkpoint
            i.e. the checkpoint has been converted to the TorchTune format using convert_llama2_to_native.py.
        - When `resume_from_checkpoint` is `True`, the checkpoint was saved using the `save_checkpoint` utility.

    If either of the above assumptions are broken, a RuntimeError will be raised.

    NOTE: Similar to :func:`torch.load`, `load_checkpoint_updated` does NOT load the state dicts or the respective recipe state.
    This should be handled by the recipe.

    Args:
        ckpt_path (str): String indicating local path to saved checkpoint file.
        resume_from_checkpoint (bool): Boolean flag indicating whether this is a fresh training run or resuming from a
            previous checkpoint.
        model (Optional[nn.Module]): Model the checkpoint will be loaded into. Only needed when `resume_from_checkpoint` is
            set to True, in which case this is used to correctly load the optimizer state. Default: `None`.
        optimizer (Optional[optim.Optimizer]): Optimizer the optimizer state checkpoint will be loaded into. Only needed
            when `resume_from_checkpoint` is set to True, in which case this is used to correctly load the optimizer state.
            Default: `None`.

    Returns:
        ckpt_dict (Dict[str, Any]): Dictionary containing loaded objects. Objects in this dictionary can be used
            to restore model, optimizer, and any other checkpointed states.

    Raises:
        RuntimeError: If `ckpt_dict` does not contain a `model` key.
        RuntimeError: If `resume_from_checkpoint` is `True` and either `model` or `optimizer` is not specified.
        RuntimeError: If `resume_from_checkpoint` is `True` and `ckpt_dict` does not contain an `optimizer` key.

    Example:
        >>> ckpt_dict = torchtune.utils.checkpoint.load_checkpoint_updated(ckpt_path, resume_from_checkpoint, model, optimizer)
        >>> model.load_state_dict(ckpt_dict["model"])
        >>> optimizer.load_state_dict(ckpt_dict["optimizer"])
    """

    ckpt_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if MODEL_KEY not in ckpt_dict:
        raise RuntimeError(
            """Expected loaded checkpoint to contain a `model` key, but it does not. Ensure checkpoint was converted to
            TorchTune's native format or saved using the `save_checkpoint` utility."""
        )

    if resume_from_checkpoint and (model is None or optimizer is None):
        raise RuntimeError(
            """Expected `model` and `optimizer` to be specified when `resume_from_checkpoint` is `True`."""
        )

    if resume_from_checkpoint and OPT_KEY not in ckpt_dict:
        raise RuntimeError(
            """Expected loaded checkpoint to contain an `optimizer` key when `resume_from_checkpoint` is `True`. Ensure
            checkpoint was saved using the `save_checkpoint` utility."""
        )

    # Transform optimizer states if using FSDP and overwrite ckpt_dict["optimizer"] with the transformed optimizer state.
    if resume_from_checkpoint:
        optim_state_dict_to_load = (
            FSDP.optim_state_dict_to_load(
                model, optimizer, ckpt_dict[OPT_KEY]
            )
            if _contains_fsdp(model)
            else ckpt_dict[OPT_KEY]
        )

        ckpt_dict[OPT_KEY] = optim_state_dict_to_load

    return ckpt_dict
