# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from itertools import chain
from typing import Callable, Dict, Optional, Set, Tuple, Type, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchtune.modules.peft.lora import (
    _lora_a_init_params,
    _lora_b_init_params,
    LoRALinear,
)

from torchtune.utils._device import _validate_device_from_env, get_device
from torchtune.utils.logging import get_logger

_log: logging.Logger = get_logger()


FSDPPolicyType: Type = Callable[[nn.Module, bool, int], bool]

_valid_distributed_single_node_nnodes = ["1:1", "1"]


def _get_sharding_strategy(strategy: str) -> ShardingStrategy:
    """Helper function to convert sharding strategy strings to ShardingStrategy enum."""
    return getattr(ShardingStrategy, strategy)


def is_distributed() -> bool:
    """Check if all environment variables required to initialize torch.distributed are set
    and distributed is properly installed. This indicates a distributed run.
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    """
    port = os.environ.get("MASTER_PORT", "")
    addr = os.environ.get("MASTER_ADDR", "")
    size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", -1))
    avlb = dist.is_available()
    return bool(port and addr and size >= 1 and rank >= 0 and avlb)


def _broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcasts a tensor from a source to all other processes.

    Args:
        tensor (torch.Tensor): Tensor to broadcast.
        src (int, optional): Source rank. Defaults to 0.

    Returns:
        torch.Tensor: Broadcasted tensor.
    """
    if dist.is_available() and dist.is_initialized():
        device = tensor.device
        if dist.get_backend() == "nccl":
            tensor = tensor.to(get_device("cuda"))
        dist.broadcast(tensor, src=src, group=None)
        return tensor.to(device)
    else:
        return tensor


def init_distributed(**kwargs: Dict) -> bool:  # noqa: DOC106, DOC109
    """Initialize torch.distributed.

    Args:
        **kwargs (Dict): Additional arguments to pass to torch.distributed.init_process_group.

    Returns:
        bool: True if torch.distributed is initialized.

    Raises:
        RuntimeError: If torch.distributed is already initialized.
    """
    if is_distributed():
        if dist.is_initialized():
            raise RuntimeError("torch.distributed already initialized.")
        dist.init_process_group(**kwargs)
        return True
    else:
        return False


def get_world_size_and_rank() -> Tuple[int, int]:
    """Function that gets the current world size (aka total number
    of ranks) and rank number of the current trainer.

    Returns:
        Tuple[int, int]: world size, rank
    """
    if dist.is_available() and dist.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    else:
        return 1, 0


def validate_no_params_on_meta_device(model: nn.Module) -> None:
    """
    Utility to validate that model has no params or buffers on meta device.
    If a meta param or buffer is found, an error indicating the param name will
    be raised.

    Args:
        model (nn.Module): model to check for meta params

    Raises:
        RuntimeError: If meta params or buffers exist in model
    """
    for n, p in chain(model.named_parameters(), model.named_buffers()):
        if p.is_meta:
            raise RuntimeError(f"Unexpected param or buffer {n} on meta device.")


def contains_fsdp(model: nn.Module) -> bool:
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


def wrap_fsdp(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    strategy: Optional[str] = None,
    auto_wrap_policy: Optional[Union[Set[Type], FSDPPolicyType]] = None,
    use_meta_device: bool = False,
    **kwargs,
) -> nn.Module:
    """Utility to setup distributed training using the torch.distributed FullyShardedDataParallel (FSDP) module.
    FSDP allows three primary types of data parallel training (these can be set under "strategy"):

    NO_SHARD:
        No sharding is done, this is standard Data Parallel training. The is typically fastest if the entire
        model and optimizer can fit on a single GPU and you just want to split the batch across ranks.
    SHARD_GRAD_OP:
        Only gradients and optimizer are sharded across all ranks. This is typically fastest when the
        model can fit on your GPU but there isn't enough room for a forward and backward pass.
    FULL_SHARD:
        All parameters are sharded across all ranks. This is necessary when even the model cannot fit on a
        single GPU.

    If using sharding, you need to define how the model is sharded. The auto_wrap_policy is a list of model layers
    and blocks that FSDP will use as shards.

    Args:
        model (nn.Module): Model to wrap for distributed training.
        device (torch.device): Device for host model.
        dtype (torch.dtype): dtype for mixed precision training. FSDP mixed precision will be
            configured to use this dtype for both computation and communication.
        strategy (Optional[str]): Sharding strategy to use. Please see
            torch.distributed.fsdp.ShardingStrategy for options. Default: "FULL_SHARD", which
            shards parameters, gradients, and optimizer states.
        auto_wrap_policy (Optional[Union[Set[Type], FSDPPolicyType]]): nn.Module types to recursively apply FSDP to.
            FSDP will wrap each instance of the specified nn.Module type in its own atomic FSDP unit.
            Alternatively, this can be a custom callable policy of type FSDPPolicyType, in which case FSDP will
            be wrapped according to the specified policy.
            Please see https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html#transformer-wrapping-policy
            for details on FSDP wrapping and writing wrapping policies.
            Default: None. In this case, FSDP is only applied to the top level module. In this
            case, entire model is unsharded during computation and memory is only saved due to
            sharding optimizer states.
        use_meta_device (bool): Set this to True if the input model has been initialized on meta device.
            If so, we will define the `reset_parameters()` method on all submodules
            to ensure FSDP properly initializes all modules on device given by `device`. Default: False
        **kwargs: additional arguments to pass to FSDP for distributed training.

    Returns:
        nn.Module: Model wrapped for distributed training

    Raises:
        RuntimeError: If environment not setup for distributed training.

    """
    if dist.is_available() and dist.is_initialized():
        if use_meta_device:
            model = prepare_model_for_fsdp_with_meta_device(model)
        if strategy is None:
            strategy = "FULL_SHARD"
        _validate_device_from_env(device)
        wrap_policy = (
            ModuleWrapPolicy(auto_wrap_policy)
            if isinstance(auto_wrap_policy, set)
            else auto_wrap_policy
        )
        mp = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        return FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            device_id=device,
            mixed_precision=None,
            sharding_strategy=_get_sharding_strategy(strategy),
            **kwargs,
        )
    else:
        raise RuntimeError(
            "Distributed environment is not setup. Please run init_distributed() first."
        )


def _dummy_reset_params(x: nn.Module) -> None:
    """
    Dummy method for patching no-op reset_parameters() when using
    FSDP with meta device.
    """
    return


def prepare_model_for_fsdp_with_meta_device(model: nn.Module) -> nn.Module:
    """
    Dynamically define reset_parameters on every submodule of the model. For LoRA models,
    ensure that the FSDP contract of reset_parameters only modifying a module's directly-owned
    parameters is satisfied. More details here: https://github.com/pytorch/pytorch/issues/104187.

    Args:
        model (nn.Module): model class to prepare for usage with FSDP and meta device.

    Returns:
        nn.Module: Model with reset_parameters defined on every submodule.
        In the case of a LoRA model, we override the default reset_parameters of nn.Linear.

    Raises:
        RuntimeError: if model contains submodule with non-callable attribute reset_parameters
    """
    for k, v in model.named_modules():
        # If the module does not have reset_parameters defined, we define
        # a no-op reset_parameters method to satisfy FSDP's contract.
        reset_params = getattr(v, "reset_parameters", None)

        if reset_params is not None and not callable(reset_params):
            raise RuntimeError(
                f"Cannot override existing reset_parameters variable for FSDP init in {k}"
            )

        if reset_params is None:
            v.reset_parameters = _dummy_reset_params.__get__(v)

        # This will define reset_parameters for LoRA weight initialization
        # directly on any LoRALinear submodules lora_a and lora_b.
        if isinstance(v, LoRALinear):
            v.lora_a.reset_parameters = _lora_a_init_params.__get__(v.lora_a)
            v.lora_b.reset_parameters = _lora_b_init_params.__get__(v.lora_b)

    return model


def lora_fsdp_wrap_policy(modules_to_wrap: Set[Type]) -> FSDPPolicyType:
    """
    A default policy for wrapping models trained with LoRA using FSDP. Specifically,
    this will wrap individual LoRA A & B submodules in their own FSDP units to
    maximize memory savings. After this is done, model will also be hierarchically wrapped
    based on nn.Module types specified in ``modules_to_wrap``. This function assumes that
    (a) LoRA's A and B matrices are the only trainable weights in the entire model, and
    (b) we have already set requires_grad = True on LoRA params.

    Args:
        modules_to_wrap (Set[Type]): nn.Module types to recursively wrap

    Returns:
        FSDPPolicyType: Wrapping policy that can be passed into ``FullyShardedDataParallel``.
    """

    def lora_wrap_fsdp(module: nn.Module, recurse: bool, **kwargs):
        if recurse:
            return True

        # Assumes lora_a and lora_b are nn.Linears that are the
        # only trainable modules in the entire network. Wraps
        # these in separate FSDP unit to work around FSDP allocating
        # extra gradient memory when wrapped with other modules.
        if hasattr(module, "weight") and module.weight.requires_grad:
            return True

        return isinstance(module, tuple(modules_to_wrap))

    return lora_wrap_fsdp
