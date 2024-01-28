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
    but it must define the ``_adapter_params()`` method.
    """

    @classmethod
    def _adapter_params(cls) -> List[str]:
        """
        Return a list of strings corresponding to the names of the adapter
        params in the model. These can be either nn.Module names or nn.Parameter names.
        E.g. if an nn.Module has adapter self.proj = nn.Linear(in_dim, out_dim), then
        either ['proj'] or ['proj.weight', 'proj.bias'] would be acceptable outputs.
        """
        pass


def _get_adapter_params(model: nn.Module) -> Dict[str, Any]:
    """
    Return the subset of parameters from a model that correspond to an adapter.
    Assumes that any adapter class has defined the _adapter_params() method.

    Args:
        model (nn.Module): Instance of model class containing some adapter params.

    Returns:
        Dict[str, Any]: the subset of model's state dict containing
        only adapter parameters.

    Raises:
        TypeError: If any of the model's submodules with _get_adapter_params
            defined return strings identifying properties with types other than
            nn.Module or nn.Parameter
    """
    adapter_params = {}
    for k, v in model.named_modules():
        if hasattr(v, "_adapter_params"):
            current_adapter_params = v._adapter_params()
            for p in current_adapter_params:
                current_params = getattr(v, p)
                prefix_key = ".".join([k, p]) if k else p
                if isinstance(current_params, nn.Parameter):
                    adapter_params.update({prefix_key: v})
                elif isinstance(current_params, nn.Module):
                    adapter_params.update(
                        {
                            ".".join([prefix_key, k1]): v1
                            for k1, v1 in getattr(v, p).state_dict().items()
                        }
                    )
                else:
                    raise TypeError(
                        f"Supported types for adapter_params are nn.Module and nn.Parameter, found {type(current_params)}"
                    )
    return adapter_params


def _get_base_model_params(model: nn.Module) -> Dict[str, Any]:
    """
    Given a model containing some adapter weights, return the subset of the model's
    parameters that correspond to the base model.
    Assumes that any adapter class has defined the _adapter_params() method.

    Args:
        model (nn.Module): Instance of model class containing some adapter params.

    Returns:
        Dict[str, Any]: the subset of adapted model's state dict containing
        only the base model's parameters.
    """
    adapter_params = _get_adapter_params(model)
    return {k: v for k, v in model.state_dict().items() if k not in adapter_params}


def _set_trainable_params(model: nn.Module) -> None:
    """
    Given a model containing some adapter weights, return the subset of the model's
    parameters that correspond to the base model.
    Assumes that any adapter class has defined the _get_adapter_params() method.

    Args:
        model (nn.Module): Instance of model class containing some adapter params.

    Returns:
        None
    """
    adapter_params = _get_adapter_params(model)
    logger.warn(
        f"Setting {adapter_params.keys()} as trainable, all other model params will be frozen."
    )
    for k, v in model.named_parameters():
        if k in adapter_params:
            v.requires_grad = True
        else:
            v.requires_grad = False
