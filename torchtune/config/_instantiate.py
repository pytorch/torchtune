# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import sys
from typing import Any, Callable, Dict, Tuple

from omegaconf import DictConfig, OmegaConf
from torchtune.config._errors import InstantiationError
from torchtune.config._utils import _get_component_from_path, _has_component


def _create_component(
    _component_: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
):
    return _component_(*args, **kwargs)


def _instantiate_node(node: Dict[str, Any], *args: Tuple[Any, ...]):
    """
    Creates the object specified in _component_ field with provided positional args
    and kwargs already merged. Raises an InstantiationError if _component_ is not specified.
    """
    if _has_component(node):
        _component_ = _get_component_from_path(node.get("_component_"))
        kwargs = {k: v for k, v in node.items() if k != "_component_"}
        return _create_component(_component_, args, kwargs)
    else:
        raise InstantiationError(
            "Cannot instantiate specified object."
            + "\nMake sure you've specified a _component_ field with a valid dotpath."
        )


def instantiate(
    config: DictConfig,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> Any:
    """
    Given a DictConfig with a _component_ field specifying the object to instantiate and
    additional fields for keyword arguments, create an instance of the specified object.
    You can use this function to create the exact instance of a torchtune object you want
    to use in your recipe using the specification from the config.

    This function also supports passing in positional args and keyword args within the
    function call. These are automatically merged with the provided config, with keyword
    args taking precedence.

    Based on Hydra's `instantiate` utility from Facebook Research:
    https://github.com/facebookresearch/hydra/blob/main/hydra/_internal/instantiate/_instantiate2.py#L148

    Args:
        config (DictConfig): a single field in the OmegaConf object parsed from the yaml file.
            This is expected to have a _component_ field specifying the path of the object
            to instantiate.
        *args (Tuple[Any, ...]): positional arguments to pass to the object to instantiate.
        **kwargs (Dict[str, Any]): keyword arguments to pass to the object to instantiate.

    Examples:
        >>> config.yaml:
        >>>     model:
        >>>       _component_: torchtune.models.llama2
        >>>       num_layers: 32
        >>>       num_heads: 32
        >>>       num_kv_heads: 32

        >>> from torchtune import config
        >>> vocab_size = 32000
        >>> # Pass in vocab size as positional argument. Since it is positioned first
        >>> # in llama2(), it must be specified first. Pass in other arguments as kwargs.
        >>> # This will return an nn.Module directly for llama2 with specified args.
        >>> model = config.instantiate(parsed_yaml.model, vocab_size, max_seq_len=4096, embed_dim=4096)

    Returns:
        Any: the instantiated object.

    Raises:
        ValueError: if config is not a DictConfig.
    """

    # Return None if config is None
    if config is None:
        return None
    if not OmegaConf.is_dict(config):
        raise ValueError(f"instantiate only supports DictConfigs, got {type(config)}")

    # Ensure local imports are able to be instantiated
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    config_copy = copy.deepcopy(config)
    config_copy._set_flag(
        flags=["allow_objects", "struct", "readonly"], values=[True, False, False]
    )
    config_copy._set_parent(config._get_parent())
    config = config_copy

    if kwargs:
        # This overwrites any repeated fields in the config with kwargs
        config = OmegaConf.merge(config, kwargs)

    # Resolve all interpolations, or references to other fields within the same config
    OmegaConf.resolve(config)

    return _instantiate_node(OmegaConf.to_object(config), *args)
