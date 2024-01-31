# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Protocol

from torch import nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            for p in current_adapter_params:
                assert (
                    p in v.state_dict()
                ), f"nn.Parameter {p} not found in {v.state_dict().keys()}"
                full_key = ".".join([k, p]) if k else p
                adapter_params.update({full_key: v.state_dict()[p]})
    return adapter_params


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
    Given a model containing some adapter weights, return the subset of the model's
    parameters that correspond to the base model.

    Args:
        model (nn.Module): Instance of model class containing some adapter params.
        adapter_params (Dict[str, Any]): State dict mapping adapter key names to their
            respective nn.Parameters (i.e. outputs of :func:`~torchtune.modules.peft.get_adapter_params`.)

    Returns:
        None
    """
    logger.warning(
        f"Setting {adapter_params.keys()} as trainable, all other model params will be frozen."
    )
    for k, v in model.named_parameters():
        v.requires_grad_(k in adapter_params)
