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
    full_model_state_dict_keys: List[str],
    lora_state_dict_keys: Optional[List[str]] = None,
    base_model_state_dict_keys: Optional[List[str]] = None,
) -> None:
    is_lora_param = lambda x: "lora" in x and any([k in x for k in lora_modules])
    for k in full_model_state_dict_keys:
        if not is_lora_param(k):
            if base_model_state_dict_keys is not None:
                assert (
                    k in base_model_state_dict_keys
                ), f"Missing non-LoRA key {k} from base model state dict"
            if lora_state_dict_keys is not None:
                assert (
                    k not in lora_state_dict_keys
                ), f"Non-LoRA key {k} found in LoRA state dict"
        else:
            if base_model_state_dict_keys is not None:
                assert (
                    k not in base_model_state_dict_keys
                ), f"LoRA key {k} found in base model state dict"
            if lora_state_dict_keys is not None:
                assert (
                    k in lora_state_dict_keys
                ), f"Missing LoRA key {k} From LoRA state dict"

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
