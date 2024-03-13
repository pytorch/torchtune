# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import functools
from typing import Any, Dict, Generator, List, Optional, Protocol

from torch import nn


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
            ["w1", "w2", "w3", "q_proj", "k_proj", "v_proj", "output_proj", "output"]
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
    is_lora_param = lambda x: any([".".join([k, "lora"]) in x for k in lora_modules])
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


def _is_eligible_for_state_dict_hook(m: nn.Module) -> bool:
    """
    Check if a module is eligible for adding/removing merge and unmerge state dict hooks.
    Currently this only supports LoRALinear weight merging.

    Args:
        m (nn.Module): Instance of model class containing some LoRA modules.

    Returns:
        bool: True if the module has the required methods for state dict hooks.
    """
    return (
        hasattr(m, "_merge_lora_weights")
        and callable(m._merge_lora_weights)
        and hasattr(m, "_unmerge_lora_weights")
        and callable(m._unmerge_lora_weights)
    )


def _register_lora_weight_merge_hooks(model: nn.Module) -> None:
    """
    Register state dict hooks for merging and unmerging LoRA weights.
    Args:
        model (nn.Module): Instance of model class containing some LoRA params.
    Returns:
        None
    Raises:
        RuntimeError: If the model already has LoRA merge state dict pre- or post-hooks.
    """
    for n, m in model.named_modules():
        if _is_eligible_for_state_dict_hook(m):
            if hasattr(m, "_merge_weight_pre_handle"):
                raise RuntimeError(
                    f"Cannot register state dict pre-hook for weight merge, {m} already has state dict weight merge pre-hook"
                )
            if hasattr(m, "_merge_weight_post_handle"):
                raise RuntimeError(
                    f"Cannot register state dict post-hook for weight merge, {m} already has state dict weight merge post-hook"
                )
            m._merge_weight_pre_handle = m.register_state_dict_pre_hook(
                m._merge_lora_weights
            )
            m._merge_weight_post_handle = m._register_state_dict_hook(
                m._unmerge_lora_weights
            )


def _unregister_lora_weight_merge_hooks(model: nn.Module) -> None:
    """
    Unregister state dict hooks for merging and unmerging LoRA weights.
    Args:
        model (nn.Module): Instance of model class containing some LoRA params.
    Returns:
        None
    Raises:
        RuntimeError: If the model does not have the expected state dict pre- or post-hooks.
    """
    for n, m in model.named_modules():
        if _is_eligible_for_state_dict_hook(m):
            if not hasattr(m, "_merge_weight_pre_handle"):
                raise RuntimeError(
                    f"Cannot unregister state dict weight merge pre-hook from {m}"
                )
            if not hasattr(m, "_merge_weight_post_handle"):
                raise RuntimeError(
                    f"Cannot unregister state dict weight merge post-hook from {m}"
                )
            m._merge_weight_pre_handle.remove()
            m._merge_weight_post_handle.remove()
            del m._merge_weight_pre_handle
            del m._merge_weight_post_handle


@contextlib.contextmanager
def merge_lora_weights_in_state_dict(model: nn.Module) -> Generator[None, None, None]:
    """
    Context manager for merging and unmerging LoRA weights in the model's state dict.
    By wrapping your `.state_dict()` call on a model containing LoRA modules in this
    context manager, you can get a state dict with the LoRA weights merged into
    the base model weights.

    Args:
        model (nn.Module): Instance of model class containing some LoRA modules.

    Returns:
        Context manager for merging and unmerging LoRA weights in the model's state dict.
    """
    _register_lora_weight_merge_hooks(model)
    try:
        yield
    finally:
        _unregister_lora_weight_merge_hooks(model)
