# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Dict, List, Optional, Protocol

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
    missing_base_model_keys: Optional[List[str]] = None,
    unexpected_base_model_keys: Optional[List[str]] = None,
    missing_lora_keys: Optional[List[str]] = None,
    unexpected_lora_keys: Optional[List[str]] = None,
) -> None:
    """
    Validate that the missing and unexpected keys for loading either
    base model weights or LoRA params into LoRA model with strict=False are as expected.

    Args:
        lora_modules (List[str]): List of LoRA modules in the model
        missing_base_model_keys (Optional[List[str]]): List of missing keys in the state
            dict for the base model.
        unexpected_base_model_keys (Optional[List[str]]): List of unexpected keys in the state
            dict for the base model.
        missing_lora_keys (Optional[List[str]]): List of missing keys in the state
            dict for the LoRA model.
        unexpected_lora_keys (Optional[List[str]]): List of unexpected keys in the state
            dict for the LoRA model.

    Returns:
        None

    Raises:
        AssertionError: If any of the missing or unexpected keys are not as expected.

    """
    if unexpected_base_model_keys:
        raise AssertionError(
            f"Unexpected keys {unexpected_base_model_keys} in base model state dict"
        )
    if unexpected_lora_keys:
        raise AssertionError(
            f"Unexpected keys {unexpected_lora_keys} in LoRA state dict"
        )
    if missing_base_model_keys:
        for x in missing_base_model_keys:
            if not any([k in x for k in lora_modules]):
                raise AssertionError(
                    f"Missing key {x} is not a LoRA module {lora_modules}"
                )
    if missing_lora_keys:
        for x in missing_lora_keys:
            if any([k in x and "lora" in x for k in lora_modules]):
                raise AssertionError(
                    f"Missing LoRA param {x} from loaded LoRA checkpoint"
                )
