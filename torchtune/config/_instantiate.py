# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Callable, Dict, Tuple, Union

from omegaconf import DictConfig, OmegaConf
from torchtune.config._utils import get_object_from_path


def has_component(node: Union[DictConfig, Dict[str, Any]]) -> bool:
    if isinstance(node, dict) or OmegaConf.is_dict(node):
        return "_component_" in node
    return False


def create_component(
    _component_: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
):
    return _component_(*args, **kwargs)


def instantiate_node(node: Union[DictConfig, Dict[str, Any]], *args: Tuple[Any, ...]):
    """
    Creates the object specified in _component_ field with provided positional args
    and kwargs already merged. Raises a ValueError if _component_ is not specified.
    """
    if has_component(node):
        _component_ = get_object_from_path(node.get("_component_"))
        kwargs = {k: v for k, v in node.items() if k != "_component_"}
        return create_component(_component_, args, kwargs)
    else:
        raise ValueError(
            "Cannot instantiate specified object."
            + "\nMake sure you've specified a _component_ field with a valid dotpath."
        )


def instantiate(
    config: Union[DictConfig, Dict[str, Any]],
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> Any:
    """
    Given a DictConfig/Dict with a _component_ field specifying the object to instantiate and
    additional fields for keyword arguments, create an instance of the specified object.
    You can use this function to create the exact instance of a TorchTune object you want
    to use in your recipe using the specification from the config.

    This function also supports passing in positional args and keyword args within the
    function call. These are automatically merged with the provided config.

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

    Args:
        config (Union[DictConfig, Dict[str, Any]]): a single field in the OmegaConf object parsed from the yaml file,
            or a plain dict. This is expected to have a _component_ field specifying the path of the object
            to instantiate.
        *args (Tuple[Any, ...]): positional arguments to pass to the object to instantiate.
        **kwargs (Dict[str, Any]): keyword arguments to pass to the object to instantiate.

    Returns:
        Any: the instantiated object.

    Raises:
        ValueError: if config is not a DictConfig or Dict.
    """

    # Return None if config is None
    if config is None:
        return None
    if not OmegaConf.is_dict(config) and not isinstance(config, dict):
        raise ValueError(
            f"instantiate only supports dicts and DictConfigs, got {type(config)}"
        )

    config_copy = copy.deepcopy(config)
    config_copy._set_flag(
        flags=["allow_objects", "struct", "readonly"], values=[True, False, False]
    )
    config_copy._set_parent(config._get_parent())
    config = config_copy

    if kwargs and OmegaConf.is_dict(config):
        config = OmegaConf.merge(config, kwargs)
    elif isinstance(config, dict):
        if kwargs:
            config.update(kwargs)
        config = OmegaConf.create(config)

    # Resolve all interpolations, or references to other fields within the same config
    OmegaConf.resolve(config)

    return instantiate_node(
        config,
        *args,
    )
