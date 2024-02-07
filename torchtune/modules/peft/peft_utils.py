# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Dict, List, Protocol, Set, Type

import torch

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


def reset_lora_params(model: nn.Module, device: torch.device) -> None:
    """
    Initializes lora parameters of a given model. This is useful
    if model is initialized on meta device and custom initialization
    needs to be run for LoRA parameters. This method is meant to be used
    in tandem with ``LoRALinear``'s ``reset_lora_parameters`` and simply
    calls this method on each instance.

    Args:
        model (nn.Module): Instance of model class containing LoRA parameters
        device (torch.device): Device to initialize LoRA parameters on.
    """
    for m in model.modules():
        if hasattr(m, "reset_lora_parameters"):
            m.reset_lora_parameters(device=device)


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
    missing_keys: List[str], unexpected_keys: List[str], lora_modules: List[str]
) -> None:
    """
    Validate that the missing and unexpected keys for loading a base model into LoRA
    model with strict=False are as expected.

    Args:
        missing_keys (List[str]): List of missing keys in the state dict.
        unexpected_keys (List[str]): List of unexpected keys in the state dict.
        lora_modules (List[str]): List of LoRA modules in the model.

    Returns:
        None

    Raises:
        AssertionError: If there are unexpected keys in the loaded state dict.
        AssertionError: If there are missing keys in the loaded state dict that are not in the LoRA modules.

    """
    for x in missing_keys:
        if not any([k in x for k in lora_modules]):
            raise AssertionError(f"Missing key {x} is not a LoRA module {lora_modules}")
    if unexpected_keys:
        raise AssertionError(f"Unexpected keys {unexpected_keys} in state dict")


def lora_fsdp_wrap_policy(modules_to_wrap: Set[Type]):
    def lora_wrap(module: nn.Module, recurse: bool, **kwargs):
        if recurse:
            return True

        # Assumes lorbea_a and lora_b are nn.Linears that are the
        # only trainable modules in the entire network. Wraps
        # these in separate FSDP unit to work around FSDP allocating
        # extra gradient memory when wrapped with other modules.
        if hasattr(module, "weight") and module.weight.requires_grad:
            return True

        return isinstance(module, tuple(modules_to_wrap))

    return lora_wrap


def lora_fsdp_init(module: nn.Module, device: torch.device):
    # Custom init for RoPE, which has buffers only
    if hasattr(module, "_init"):
        module._init(device=device)
    # Skip init of modules that already have params on non-meta device
    if all([not p.is_meta for p in module.parameters()]):
        return
    else:
        # Brings params to device with empty data. data will be
        # overwriten when loading in checkpoint.
        module.to_empty(device=device, recurse=False)
