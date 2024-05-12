# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from itertools import chain
from typing import Callable, Dict, Set, Tuple, Type

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchtune import modules
from torchtune.modules.peft.lora import (
    _lora_a_init_params,
    _lora_b_init_params,
    LoRALinear,
)

from torchtune.utils._device import get_device
from torchtune.utils.logging import get_logger

_log: logging.Logger = get_logger()


FSDPPolicyType: Type = Callable[[nn.Module, bool, int], bool]

FSDPPolicyType.__doc__ = """

A datatype for a function that can be used as an FSDP wrapping policy.
In particular, this type denotes a function that can accept an nn.Module, a boolean flag, and an integer
and return a boolean indicating whether the module should be wrapped with FSDP. Objects of this type can
be directly passed into PyTorch FSDP's ``auto_wrap_policy`` argument to specify how FSDP wraps submodules.

The below function serves as an example of creating and returning a function that obeys the contract of
``FSDPPolicyType``::

    def get_fsdp_policy(module: nn.Module, modules_to_wrap: Set[Type], min_num_params: int):

        def my_fsdp_policy(module: nn.Module, modules_to_wrap: Set[Type], recurse: bool, min_num_params: int) -> bool:
            if recurse:
                return True
            # Wrap layers that are of type in ``modules_to_wrap`` and layers with more than min_num_params

            return isinstance(module, tuple(modules_to_wrap)) or sum(p.numel() for p in module.parameters()) > 1000

        return functools.partial(my_fsdp_policy, modules_to_wrap=modules_to_wrap)

Please see documentation of ``auto_wrap_policy`` at https://pytorch.org/docs/stable/fsdp.html for additional details.

"""

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
        FSDPPolicyType: Wrapping policy that can be passed into ``FullyShardedDataParallel``. Please see
        documentation for :const:`~torchtune.utils.FSDPPolicyType` for additional details.
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


def get_full_finetune_fsdp_wrap_policy(
    memory_efficient_fsdp_wrap: bool, modules_to_wrap: Set[Type]
) -> FSDPPolicyType:
    """
    Retrieves an FSDP wrapping policy based on the specified flags ``memory_efficient_fsdp_wrap`` and
    ``modules_to_wrap``. Specifically, if ``memory_efficient_fsdp_wrap`` is set to ``True``, the returned
    policy will wrap the model's token embedding and output projection in addition to the modules specified
    to maximize memory savings.

    Args:
        memory_efficient_fsdp_wrap (bool): If ``True``, will also wrap embedding and output projection layers with FSDP.
        modules_to_wrap (Set[Type]): Set of module types to wrap.

    Note:
        ``memory_efficient_fsdp_wrap`` memory improvements have currently only been verified on llama3 workloads,
        to provide ~15% memory improvement (when used alongside AC memory efficient wrapping). Other workloads
        have not been verified and may not see the same improvements.
    Returns:
        FSDPPolicyType: Wrapping policy that can be passed into ``FullyShardedDataParallel`` as the ``auto_wrap_policy``
            argument. Please see documentation for const:`~torchtune.utils.FSDPPolicyType` for additional details.
    """
    if memory_efficient_fsdp_wrap:
        return _memory_efficient_wrap_policy(modules_to_wrap=modules_to_wrap)
    else:
        return ModuleWrapPolicy(modules_to_wrap)


def _memory_efficient_wrap_policy(modules_to_wrap: Set[Type]) -> FSDPPolicyType:
    """
    A default policy for memory efficient wrapping for full finetuning using FSDP. Specifically,
    this will wrap the model's token embedding and output projection into their own FSDP units to
    maximize memory savings. This helps especially if these layers are particularly large,
    such as due to a large embedding size.
    After this is done, model will also be hierarchically wrapped
    based on nn.Module types specified in ``modules_to_wrap``. This function assumes that the
    input model has an attribute ``output`` that is a nn.Linear which is the model's output projection.
    Args:
        modules_to_wrap (Set[Type]): nn.Module types to recursively wrap
    Returns:
        FSDPPolicyType: Wrapping policy that can be passed into ``FullyShardedDataParallel``.
    """
    modules_to_wrap.add(torch.nn.Embedding)

    def llama3_wrap(module: nn.Module, recurse: bool, **kwargs):
        # Label that output_proj should be wrapped individually.
        if isinstance(module, modules.TransformerDecoder):
            module.output._wrap = True
        if recurse:
            return True

        # Wrap output_proj individually.
        if getattr(module, "_wrap", False):
            return True

        return isinstance(module, tuple(modules_to_wrap))

    return llama3_wrap
