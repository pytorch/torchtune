from typing import Any, Dict, Optional

import torch


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
        raise ValueError(
            "Expected `ckpt_dict` to contain a `model` key, but it does not."
        )
    model_state_dict = ckpt_dict["model"].state_dict()
    if "optimizer" in ckpt_dict:
        if _contains_fsdp(ckpt_dict["model"]):
            optimizer_state_dict = FSDP.optim_state_dict(
                ckpt_dict["model"], ckpt_dict["optimizer"]
            )
        else:
            optimizer_state_dict = ckpt_dict["optimizer"].state_dict()

    ckpt_dict["model"] = model_state_dict
    if "optimizer" in ckpt_dict:
        ckpt_dict["optimizer"] = optimizer_state_dict

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        torch.save(ckpt_dict, output_loc)


def load_checkpoint(
    ckpt_dict: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """
    Loads a checkpoint from `ckpt_dict` into `model` and optionally `optimizer`. This function is meant to be used in tandem with
    `save_checkpoint` and assumes the checkpoint was saved with `save_checkpoint` and subsequently loaded via `torch.load`.

    Args:
        ckpt_dict (Dict[str, Any]): Dictionary containing the checkpoint to be saved. Must have at least `model` key.
        model (torch.nn.Module): Model to load the checkpoint into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load the checkpoint into. If not specified, "optimizer" key in `ckpt_dict`
            will be ignored, if present. Default: `None`.
    """

    if "model" not in ckpt_dict:
        raise ValueError(
            "Expected `ckpt_dict` to contain a `model` key, but it does not."
        )
    model.load_state_dict(ckpt_dict["model"])

    if optimizer is not None:
        if "optimizer" not in ckpt_dict:
            raise ValueError(
                "Expected `ckpt_dict` to contain an `optimizer` key, but it does not."
            )
        if _contains_fsdp(model):
            # Preprocess optim_state_dict for FSDP.
            FSDP.optim_state_dict_to_load(ckpt_dict["optimizer"], model, optimizer)

        optimizer.load_state_dict(ckpt_dict["optimizer"])
