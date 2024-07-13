# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from itertools import chain
from typing import Any, Callable, Dict, Set, Tuple, Type

import torch
import torch.distributed as dist
from torch import nn

from torch.distributed._tensor import distribute_tensor, DTensor
from torch.distributed.checkpoint.state_dict import _init_optim_state
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.optim import Optimizer
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


def set_torch_num_threads() -> None:
    """
    Sets the number of threads used by torch to utilize all physical CPU
    cores for intra-op parallelism. Currently, this function sets num_threads
    to be the number of physical CPU cores divided by the number of GPUs as we
    use one process per GPU, and this avoids CPU oversubscription. Note that this is
    currently a rough approximation, and doesn't take into account environments where
    things like CPU affinity is set.
    """
    num_threads = os.cpu_count() // (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    torch.set_num_threads(num_threads)
    _log.info(f"Set intra op parallelism no. of threads to {num_threads}")


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


def load_from_full_model_state_dict(
    model: "FSDPModule",
    full_sd: Dict[str, Any],
    device: torch.device,
    is_rank_zero: bool,
):
    """
    Converting full state dict into a sharded state dict
    and loading it into FSDP model
    - 'full' means plain tensor
    - 'sharded' means `DTensor` where reach rank has a shard of the plain tensor
    - `is_rank_zero` matters if only rank 0 pass in non-empty `full_sd` and
       we need to broadcast from rank 0
    """
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    for param_name, full_tensor in full_sd.items():
        sharded_meta_param = meta_sharded_sd.get(param_name)
        # `.to(dtype)` ensures same dtype when `assign=True`
        sharded_tensor = distribute_tensor(
            full_tensor.to(sharded_meta_param.dtype),
            sharded_meta_param.device_mesh,
            sharded_meta_param.placements,
        )
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)
    # choose `assign=True` since we cannot call `copy_` on meta tensor
    model.load_state_dict(sharded_sd, strict=False, assign=True)


def get_full_model_state_dict(
    model: "FSDPModule",
    is_rank_zero: bool,
) -> Dict[str, Any]:
    """
    Converting sharded state dict into a full state dict on cpu
    Returning non-empty result on rank0 to avoid peaking cpu memory
    """
    sharded_sd = model.state_dict()
    cpu_state_dict = {}
    for param_name, sharded_param in sharded_sd.items():
        full_param = sharded_param.full_tensor()
        if is_rank_zero:
            cpu_state_dict[param_name] = full_param.cpu()
        else:
            del full_param
    return cpu_state_dict


def get_full_optimizer_state_dict(
    opt: Optimizer,
    is_rank_zero: bool,
) -> Dict[str, Any]:
    """
    Converting optimizer state from sharded to full
    For example, "exp_avg" in AdamW is `DTensor`,
    "exp_avg.full_tensor()" converts it to plain tensor on rank 0
    Returning non-empty cpu state dict on rank 0
    """
    sharded_sd = opt.state_dict()
    sharded_state = sharded_sd["state"]
    full_state = {}
    for group_id, sharded_group in sharded_state.items():
        group_state = {}
        for attr, sharded_tensor in sharded_group.items():
            if isinstance(sharded_tensor, DTensor):
                # "exp_avg" in AdamW is `DTensor`
                full_tensor = sharded_tensor.full_tensor()
            else:
                # "step" in AdamW is plain tensor
                full_tensor = sharded_tensor
            if is_rank_zero:
                group_state[attr] = full_tensor.cpu()
            else:
                del full_tensor
        if is_rank_zero:
            full_state[group_id] = group_state
        else:
            del group_state
    if is_rank_zero:
        return {
            "param_groups": sharded_sd["param_groups"],
            "state": full_state,
        }
    else:
        return {}


def load_from_full_optimizer_state_dict(
    opt: Optimizer,
    full_sd: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Converting full optimizer state to sharded state dict
    and loading it into optimizer
    """
    PARAMS = "params"  # noqa: N806
    _init_optim_state(opt)
    param_groups = opt.state_dict()["param_groups"]
    state = opt.state_dict()["state"]

    full_param_groups = full_sd["param_groups"]
    full_state = full_sd["state"]

    for param_group, full_param_group in zip(param_groups, full_param_groups):
        for key, value in full_param_group.items():
            if key == PARAMS:
                continue
            param_group[key] = value
        for pid, full_pid in zip(param_group[PARAMS], full_param_group[PARAMS]):
            if pid not in state:
                continue
            param_state = state[pid]
            full_param_state = full_state[full_pid]
            for attr, full_tensor in full_param_state.items():
                sharded_tensor = param_state[attr]
                if isinstance(sharded_tensor, DTensor):
                    # exp_avg is DTensor
                    param_state[attr] = distribute_tensor(
                        full_tensor,
                        sharded_tensor.device_mesh,
                        sharded_tensor.placements,
                    )
                else:
                    # step is plain tensor
                    param_state[attr] = full_tensor
    opt.load_state_dict(
        {
            "param_groups": param_groups,
            "state": state,
        }
    )


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
