# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from itertools import chain
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed._tensor import distribute_tensor, DTensor
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
from torch.distributed.checkpoint.state_dict import _init_optim_state
from torch.distributed.fsdp import ShardingStrategy
from torch.optim import Optimizer
from torchao.dtypes.nf4tensor import NF4Tensor, to_nf4
from torchtune.modules import TransformerDecoder
from torchtune.utils import get_logger

from torchtune.utils._device import get_device

_log: logging.Logger = get_logger()


_valid_distributed_single_node_nnodes = ["1:1", "1"]


def _get_sharding_strategy(strategy: str) -> ShardingStrategy:
    """Helper function to convert sharding strategy strings to ShardingStrategy enum."""
    return getattr(ShardingStrategy, strategy)


def is_distributed() -> bool:
    """Check if all environment variables required to initialize torch.distributed are set
    and distributed is properly installed. This indicates a distributed run.
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization

    Checks the following conditions:

    * torch.distributed is available
    * master port and master address environment variables are set
    * world size is >1
    * rank environment variable is set

    Returns:
        bool: True if all of the above conditions hold, False otherwise.
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
        tensor (torch.Tensor): torch.Tensor to broadcast.
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


def init_distributed(**kwargs: Dict[str, Any]) -> bool:
    """Initialize process group required for ``torch.distributed``.

    Args:
        **kwargs (Dict[str, Any]): Additional arguments to pass to torch.distributed.init_process_group.

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
    of ranks) and rank number of the current process in the default process group.

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


def load_from_full_model_state_dict(
    model: "FSDPModule",  # noqa
    full_sd: Dict[str, Any],
    device: torch.device,
    is_rank_zero: bool,
    strict: bool = False,
    cpu_offload: bool = False,
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
        full_tensor = full_tensor.to(sharded_meta_param.dtype).to(device)
        if hasattr(sharded_meta_param, "_local_tensor") and isinstance(
            sharded_meta_param._local_tensor, NF4Tensor
        ):
            block_size = sharded_meta_param._local_tensor.block_size
            scaler_block_size = sharded_meta_param._local_tensor.scaler_block_size
            full_tensor = to_nf4(
                full_tensor, block_size=block_size, scaler_block_size=scaler_block_size
            )
            # replicating logic from `_fsdp_param.py`` `_init_sharded_param`
            # otherwise `distribute_tensor(DTensor(local=NF4))`
            # requires dispatching `c10d.scatter_``
            # long-term solution is `swap_tensor`
            mesh = sharded_meta_param.device_mesh
            if mesh.ndim > 1:
                raise NotImplementedError(f"only support 1D FSDP but got {mesh.ndim=}")
            shard_mesh_dim = 0
            shard_world_size = mesh.size(shard_mesh_dim)
            shard_rank = cast(
                torch.distributed.ProcessGroup, mesh.get_group(shard_mesh_dim)
            ).rank()
            chunk = list(torch.chunk(full_tensor, shard_world_size, dim=0))[shard_rank]
            sharded_param = full_tensor.new_zeros(chunk.size())
            sharded_param[: chunk.size(0)].copy_(chunk)

            # TODO: change to from_local API (need to add view support for NF4)
            sharded_tensor = DTensor(
                local_tensor=sharded_param,
                spec=DTensorSpec(
                    mesh=sharded_meta_param.device_mesh,
                    placements=sharded_meta_param.placements,
                    tensor_meta=TensorMeta(
                        shape=sharded_meta_param.size(),
                        dtype=sharded_meta_param.dtype,
                        stride=sharded_meta_param.stride(),
                    ),
                ),
                requires_grad=sharded_meta_param.requires_grad,
            )

        elif not hasattr(sharded_meta_param, "device_mesh"):
            # In cases where parts of the model aren't sharded, some parameters will be plain tensors
            sharded_tensor = full_tensor
        else:
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
        if cpu_offload:
            sharded_tensor = sharded_tensor.cpu()
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)
    # choose `assign=True` since we cannot call `copy_` on meta tensor
    return model.load_state_dict(sharded_sd, strict=strict, assign=True)


def _gather_nf4_tensor(sharded_param: nn.Parameter) -> nn.Parameter:
    """
    Manually gather NF4Tensor parameter since it does not support all_gather
    """
    mesh = sharded_param.device_mesh
    nf4_tensor = sharded_param._local_tensor
    quant_params, metadata = nf4_tensor.fsdp_pre_all_gather(mesh)
    full_quant_params = []
    for quant_param in quant_params:
        d0, *dn = quant_param.shape
        shape = (d0 * mesh.get_group().size(), *dn)
        full_quant_param = torch.empty(
            shape, device=quant_param.device, dtype=quant_param.dtype
        )
        dist.all_gather_into_tensor(
            full_quant_param, quant_param, mesh.get_group(), async_op=False
        )
        full_quant_params.append(full_quant_param)
    full_param, _ = nf4_tensor.fsdp_post_all_gather(
        full_quant_params, metadata, nf4_tensor.dtype
    )
    return full_param


def gather_cpu_state_dict(
    sharded_sd: Dict[str, DTensor],  # noqa
    is_rank_zero: bool,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Converting sharded state dict into a full state dict on CPU
    Returning non-empty result only on rank0 to avoid peaking CPU memory

    Args:
        sharded_sd (Dict[str, DTensor]): Sharded state dict of DTensors
        is_rank_zero (bool): flag to check if the process is on rank 0
        device (Optional[torch.device]): device to use for sharded tensors. Default: None

    Returns:
        Dict[str, Any]: State dict on CPU
    """
    cpu_state_dict = {}
    for param_name, param in sharded_sd.items():
        if param.is_cpu:
            # Move back to device if offloaded to CPU
            param = param.to(device)
        if hasattr(param, "_local_tensor"):
            if isinstance(param._local_tensor, NF4Tensor):
                param = _gather_nf4_tensor(param)
            else:
                # Gather DTensor
                param = param.full_tensor()
        if isinstance(param, NF4Tensor):
            # upcasting NF4 to original dtype
            param = param.to(param.dtype)
        if is_rank_zero:
            cpu_state_dict[param_name] = param.cpu()
        torch.distributed.barrier()
    return cpu_state_dict


def get_full_optimizer_state_dict(
    opt: Optimizer,
    is_rank_zero: bool,
    device: Optional[torch.device] = None,
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
            # without this, it may hang forever for +70B models.
            torch.distributed.barrier()
            # "exp_avg" in AdamW is `DTensor`
            if isinstance(sharded_tensor, DTensor):
                if sharded_tensor.is_cpu:
                    assert device is not None and device.type == "cuda", (
                        f"Expect cuda but got device={device}. "
                        "Please call get_full_optimizer_state_dict(..., device=self._device),"
                        " so DTensor can communicate over NCCL."
                    )
                    sharded_tensor = sharded_tensor.to(device)
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


def get_shard_conditions(
    name: str,
    module: nn.Module,
    names_to_match: Optional[List[str]] = None,
    *args,
    **kwargs,
) -> bool:
    """
    Returs True for layers named {}.layers.i or layers that exactly match names_to_match, otherwise,
    returns False. This is a helper function for sharding a model with FSDP.
    In :func:`~torchtune.training.shard_model`, we iterate over the model's named modules
    and apply fully_shard using this condition.

    As part of our sharding strategy, we want each layer to be sharded separately, as this is
    generally efficient. We may also want to shard certain modules that are not layers, such as
    the embedding module.

    #TODO: a more robust way would be to shard on the module type, not the name.

    Args:
        name (str): Name of the module.
        module (nn.Module): Module to be sharded.
        names_to_match (Optional[List[str]]): List of names to match, if any.
        *args: Variable length argument list to be passed to the Embedding module.
        **kwargs: Arbitrary keyword arguments to be passed to the Embedding module.

    Returns:
        bool: True if the module name matches the condition, False otherwise.

    Examples:
        >>> names_to_match = ["embedding"]
        >>> layer_names = ["layers.0", "decoder.layers.1", "encoder.layers.2.attention",
            "my_wrapper.layer.1.something", "embedding"]
        >>> matches = []
        >>> for name in layer_names:
        >>>     if shard_condition_is_layer_or_match(name, None): matches.append(name)
        >>> print(matches)
        >>> ["layers.0", "decoder.layers.1", "embedding"]
    """
    if names_to_match and name in names_to_match:
        return True

    name_list = name.split(".")
    if len(name_list) >= 2:
        return name_list[-2] == "layers" and str.isdigit(name_list[-1])

    return False


def shard_model(
    model: TransformerDecoder,
    shard_conditions: List[Callable[[str, nn.Module], bool]],
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
) -> None:
    """
    Utility to shard a model with FSDP using the PyTorch Distributed fully_shard API.

    This method will over the model's named modules from the bottom-up and apply shard modules
    based on whether they meet any of the criteria from shard_conditions.

    Args:
        model (TransformerDecoder): Model to shard with FSDP.
        shard_conditions (List[Callable[[str, nn.Module], bool]]): A list of functions to determine
            which modules to shard with FSDP. Each function should take module name (relative to root)
            and the module itself, returning True if FSDP should shard the module and False otherwise.
            If any of shard_conditions return True for a given module, it will be sharded by FSDP.
        cpu_offload (bool): If set to True, FSDP will offload parameters, gradients, and optimizer
            states to CPU.
        reshard_after_forward (bool): Whether to reshard parameters and buffers after
            the forward pass. Setting this to True corresponds to the FULL_SHARD sharding strategy
            from FSDP1, while setting it to False corresponds to the SHARD_GRAD_OP sharding strategy.

    Raises:
        ValueError: If no layer modules were sharded, indicating that no shard_condition was triggered.
    """
    fsdp_kwargs = {"reshard_after_forward": reshard_after_forward}
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    # Shard the model with FSDP, iterating in reverse to start with
    # lowest-level modules first
    num_layers_sharded = 0
    for n, m in reversed(list(model.named_modules())):
        if any([shard_condition(n, m) for shard_condition in shard_conditions]):
            fully_shard(m, **fsdp_kwargs)
            num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError(
            "No layer modules were sharded. Please check if shard conditions are working as expected."
        )

    # Finally shard the entire model to account for any stragglers
    fully_shard(model, **fsdp_kwargs)
