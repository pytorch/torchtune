# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Dict, List, Optional, Protocol, Set, Type

import torch

from torch import nn

from torchtune.utils.distributed import FSDPPolicyType


class AdapterModule(Protocol):
    """
    Interface for an nn.Module containing adapter weights.
    Note that an adapter module does not have to explicitly implement this protocol,
    but it must define the ``adapter_params(self)`` method.
    """

    def adapter_params(self) -> List[str]:
        """
        Return a list of strings corresponding to the names of the nn.Parameters in
        the model coming from the adapter.
        E.g. if an nn.Module has adapter ``self.proj = nn.Linear(in_dim, out_dim)``,
        then adapter_params should return ``['proj.weight', 'proj.bias']``.

        See LoRALinear's :func:`~torchtune.modules.peft.LoRALinear.adapter_params` for an example.
        """
        pass


@functools.lru_cache()
def get_adapter_params(model: nn.Module) -> Dict[str, Any]:
    """
    Return the subset of parameters from a model that correspond to an adapter.
    Assumes that any adapter class has defined the
    :func:`~torchtune.modules.peft.AdapterModule.adapter_params` method.

    Args:
        model (nn.Module): Instance of model class containing some adapter params.

    Returns:
        Dict[str, Any]: the subset of model's state dict containing
        only adapter parameters.

    """
    adapter_params = {}
    for k, v in model.named_modules():
        if hasattr(v, "adapter_params") and callable(v.adapter_params):
            current_adapter_params = v.adapter_params()
            for n, p in v.named_parameters(recurse=True):
                if n in current_adapter_params:
                    full_key = f"{k}.{n}" if k else n
                    adapter_params.update({full_key: p})
                    current_adapter_params.remove(n)
            assert (
                current_adapter_params == []
            ), f"Adapter params {current_adapter_params} not converted"
    return adapter_params


@functools.lru_cache()
def _get_base_model_params(model: nn.Module) -> Dict[str, Any]:
    """
    Given a model containing some adapter weights, return the subset of the model's
    parameters that correspond to the base model. Assumes that any adapter class has
    defined the :func:`~torchtune.modules.peft.AdapterModule.adapter_params` method.

    Args:
        model (nn.Module): Instance of model class containing some adapter params.

    Returns:
        Dict[str, Any]: the subset of adapted model's state dict containing
        only the base model's parameters.
    """
    adapter_params = get_adapter_params(model)
    return {k: v for k, v in model.state_dict().items() if k not in adapter_params}


def set_trainable_params(model: nn.Module, adapter_params: Dict[str, Any]) -> None:
    """
    Set trainable parameters for an nn.Module based on a state dict of adapter parameters.

    Args:
        model (nn.Module): Instance of model class containing some adapter params.
        adapter_params (Dict[str, Any]): State dict mapping adapter key names to their
            respective nn.Parameters (i.e. outputs of :func:`~torchtune.modules.peft.get_adapter_params`.)

    Returns:
        None
    """
    for k, v in model.named_parameters():
        v.requires_grad_(k in adapter_params)


def validate_state_dict_for_lora(
    *,
    lora_modules: List[str],
    full_model_state_dict_keys: List[str],
    lora_state_dict_keys: Optional[List[str]] = None,
    base_model_state_dict_keys: Optional[List[str]] = None,
) -> None:
    """
    Validate that the state dict keys for a LoRA model are as expected.

    (1) If lora_state_dict_keys are passed, this function will confirm that they match exactly the
        LoRA param names from the full model (as determined by lora_modules).
    (2) If base_model_state_dict_keys are passed, this function will confirm that they are exactly the
        complement of the LoRA param names from the full model.
    (3) If both lora_state_dict_keys and base_model_state_dict_keys are passed, this function will
        confirm that the full model's params are exactly their disjoint union.

    Args:
        lora_modules (List[str]): List of LoRA modules in the model. Should be a subset of
            ["w1", "w2", "w3", "q_proj", "k_proj", "v_proj", "output_proj"]
        full_model_state_dict_keys (List[str]): List of keys in the full model state dict.
        lora_state_dict_keys (Optional[List[str]]): List of keys in the LoRA state dict.
            If none, LoRA state dict keys will not be validated.
        base_model_state_dict_keys (Optional[List[str]]): List of keys in the base model state dict.
            If none, base model keys will not be validated.

    Returns:
        None

    Raises:
        AssertionError: If base model state dict is missing any non-LoRA params from the full model.
        AssertionError: If LoRA state dict is missing any LoRA params from the full model.
        AssertionError: If base model state dict has any LoRA params.
        AssertionError: If LoRA state dict has any non-LoRA params.
        AssertionError: If base model and LoRA state dicts have overlapping keys.
        AssertionError: If full model state dict is missing keys from either base model or LoRA state dict.

    """
    is_lora_param = lambda x: "lora" in x and any([k in x for k in lora_modules])
    for k in full_model_state_dict_keys:
        if not is_lora_param(k):
            if base_model_state_dict_keys is not None:
                if k not in base_model_state_dict_keys:
                    raise AssertionError(
                        f"Missing non-LoRA key {k} from base model state dict"
                    )
            if lora_state_dict_keys is not None:
                if k in lora_state_dict_keys:
                    raise AssertionError(f"Non-LoRA key {k} found in LoRA state dict")
        else:
            if base_model_state_dict_keys is not None:
                if k in base_model_state_dict_keys:
                    raise AssertionError(f"LoRA key {k} found in base model state dict")
            if lora_state_dict_keys is not None:
                if k not in lora_state_dict_keys:
                    raise AssertionError(f"Missing LoRA key {k} From LoRA state dict")

    # Full model is disjoint union of base model and LoRA weights
    if lora_state_dict_keys is not None and base_model_state_dict_keys is not None:
        combined_state_dict_keys = set(lora_state_dict_keys).union(
            base_model_state_dict_keys
        )
        shared_state_dict_keys = set(lora_state_dict_keys).intersection(
            base_model_state_dict_keys
        )
        assert (
            shared_state_dict_keys == set()
        ), "Base model and LoRA state dict have overlapping keys"
        assert combined_state_dict_keys == set(
            full_model_state_dict_keys
        ), "Extra keys not present in full model"


def lora_fsdp_wrap_policy(modules_to_wrap: Set[Type]) -> FSDPPolicyType:
    """
    A default policy for wrapping models trained with LoRA in FSDP. Specifically,
    this will wrap individual LoRA a & b submodules in their own FSDP units to
    maximize memory savings. After this is done, model will also be hierarchically wrapped
    based on nn.Module types specified in ``modules_to_wrap``.

    Args:
        modules_to_wrap (Set[Type]): nn.Module types to recursively wrap

    Returns:
        FSDPPolicyType: Wrapping policy that can be passed into ``FullyShardedDataParallel``.
    """

    def lora_wrap(module: nn.Module, recurse: bool, **kwargs):
        if recurse:
            return True

        # Assumes lora_a and lora_b are nn.Linears that are the
        # only trainable modules in the entire network. Wraps
        # these in separate FSDP unit to work around FSDP allocating
        # extra gradient memory when wrapped with other modules.
        if hasattr(module, "weight") and module.weight.requires_grad:
            return True

        return isinstance(module, tuple(modules_to_wrap))

    return lora_wrap


def lora_fsdp_init(module: nn.Module, device: torch.device) -> None:
    """
    A function to specific modules within a LoRA model wrapped in FSDP that was
    initially created on the meta device. This function is meant to be
    passed as the ``param_init_fn`` arg into ``FullyShardedDataParallel``.
    This function specially handles details such as manually initializing
    ``RotaryPositionalEmbeddings`` that have custom initialization schemes.

    Args:
        module (nn.Module): module to run initialization for
        device (torch.device): device parameters should be initialized on.
    """
    # Custom init for RoPE, which has buffers only
    if hasattr(module, "_rope_init"):
        module._rope_init(device=device)
    # Skip init of modules that already have params on non-meta device. This is
    # to avoid re-initialization of parameters that have already been explicitly
    # initialized by the user before wrapping with FSDP. In particular, for LoRA,
    # LoRA a & b matrices are pre-initialized before wrapping with FSDP.
    if all([not p.is_meta for p in module.parameters()]):
        return
    else:
        # Brings params to device with empty data. data will be
        # overwriten when loading in checkpoint.
        module.to_empty(device=device, recurse=False)


def register_lora_weight_merge_hooks(model: nn.Module):
    for n, m in model.named_modules():
        # TODO: less verbose check
        if (
            hasattr(m, "merge_lora_weights")
            and callable(m.merge_lora_weights)
            and hasattr(m, "unmerge_lora_weights")
            and callable(m.unmerge_lora_weights)
        ):
            # Add check these don't already exist
            m.pre_handle = m.register_state_dict_pre_hook(m.merge_lora_weights)
            m.post_handle = m._register_state_dict_hook(m.unmerge_lora_weights)


def unregister_lora_weight_merge_hooks(model: nn.Module):
    for n, m in model.named_modules():
        if (
            hasattr(m, "merge_lora_weights")
            and callable(m.merge_lora_weights)
            and hasattr(m, "unmerge_lora_weights")
            and callable(m.unmerge_lora_weights)
        ):
            # TODO: add some assertion here
            m.pre_handle.remove()
            m.post_handle.remove()


# def lora_save_checkpoint_weight_merge_decorator(dec, condition):
#     def decorator(func):
#         if not condition:
#             # Return the function unchanged, not decorated.
#             return func
#         return dec(func)
#     return decorator

# def register_and_unregister_lora_merge_hooks(model: nn.Module, merge_lora_weights: bool = False):
# # def decorator(func: Callable):
#     def wrapper(*args, **kwargs):
#         if merge_lora_weights:
#             register_lora_weight_merge_hooks(model)
#             result = func(*args, **kwargs)
#             unregister_lora_weight_merge_hooks(model)
#         else:
#             result = func(*args, **kwargs)
#         return result
#     return wrapper


# def lora_weight_merge_pre_hook(model: nn.Module):
#     for n, m in model.named_modules():
#         if isinstance(m, LoRALinear):
#             m.post_handle = m._register_state_dict_hook(m.merge_lora_weights)

# def lora_weight_merge_post_hook(model: nn.Module):
#     for n, m in model.named_modules():
#         if isinstance(m, LoRALinear):
#             m.unmerge_lora_weights()

# def register_and_unregister_lora_merge_hooks(model: nn.Module, fn: Callable, *args, **kwargs):
#     pre_handle = model._register_pre_state_dict_hook(lora_weight_merge_pre_hook)
#     post_handle = model._register_state_dict_hook(lora_weight_merge_post_hook)
#     fn(model, *args, **kwargs)
#     pre_handle.remove()
#     post_handle.remove()

# def register_and_unregister_lora_merge_hooks(model: nn.Module, merge_lora_weights: bool = False):
# def decorator(func: Callable):
#     def wrapper(*args, **kwargs):
#         if do_decorate:
#             pre_handle = model._register_pre_state_dict_hook(lora_weight_merge_pre_hook)
#             post_handle = model._register_state_dict_hook(lora_weight_merge_post_hook)
#             result = func(*args, **kwargs)
#             pre_handle.remove()
#             post_handle.remove()
#         else:
#             result = func(*args, **kwargs)
#         return result
#     return wrapper


# def wrap_for_lora_weight_merging(fn: Callable, model: nn.Module):
#     for n, m in model.named_modules():
#         if isinstance(m, LoRALinear):
#             pre_handle = m._register_pre_state_dict_hook(m.pre_state_dict_hook)
#             post_handle = m._register_state_dict_hook(m.post_state_dict_hook)
