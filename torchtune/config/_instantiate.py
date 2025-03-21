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
from torchtune.config._utils import _get_component_from_path


def _create_component(
    _component_: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Create an instance of a component with given arguments."""
    return _component_(*args, **kwargs)


def _instantiate_node(config_dict: Dict[str, Any], *args: Any) -> Any:
    """
    Instantiate a component from a config dictionary.

    If the dictionary has a '_component_' field, retrieve the component, process
    any nested arguments, and create the object with the given positional args.

    Args:
        config_dict (Dict[str, Any]): Config dictionary with '_component_' and arguments.
        *args (Any): Positional arguments for the component.

    Returns:
        Any: The instantiated object.

    Examples:
        >>> class Spice:
        >>>     def __init__(self, heat_level):
        >>>         self.heat_level = heat_level
        >>> class Food:
        >>>     def __init__(self, seed, ingredient):
        >>>         self.seed = seed
        >>>         self.ingredient = ingredient
        >>> config_dict = {'_component_': 'Food', 'seed': 42,
        >>>                'ingredient': {'_component_': 'Spice', 'heat_level': 5}}
        >>> food = _instantiate_node(config_dict)
        >>> print(food.seed)  # 42
        >>> print(food.ingredient.heat_level)  # 5

    Raises:
        InstantiationError: If '_component_' is missing.
    """
    if "_component_" in config_dict:
        _component_ = _get_component_from_path(config_dict["_component_"])
        kwargs = {
            k: _instantiate_nested(v)
            for k, v in config_dict.items()
            if k != "_component_"
        }
        return _create_component(_component_, args, kwargs)
    raise InstantiationError("Cannot instantiate: '_component_' field is missing.")


def _instantiate_nested(obj: Any) -> Any:
    """
    Processes dictionaries and lists to recursively instantiate any nested '_component_' fields.

    Args:
        obj (Any): Object to process (dict, list, or other).

    Returns:
        Any: Object with nested components instantiated.
    """
    if isinstance(obj, dict):
        if "_component_" in obj:
            config = OmegaConf.create(obj)
            return instantiate(config)
        return {k: _instantiate_nested(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_instantiate_nested(item) for item in obj]
    return obj


def instantiate(
    config: DictConfig,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Instantiate a component from a configuration, recursively handling nested components.

    Given a DictConfig with a '_component_' field specifying the object to instantiate and
    additional fields as keyword arguments, create an instance of the specified object.
    Positional and keyword arguments passed in the call are merged with the config, with
    keyword arguments taking precedence.

    Based on Hydra's `instantiate` utility.

    Args:
        config (DictConfig): Configuration with '_component_' and optional arguments.
        *args (Any): Positional arguments for the component.
        **kwargs (Any): Keyword arguments to override or add to the config.

    Returns:
        Any: The instantiated object, or None if config is None.

    Examples:
        >>> class Spice:
        >>>     def __init__(self, heat_level):
        >>>         self.heat_level = heat_level
        >>> class Food:
        >>>     def __init__(self, seed, ingredient):
        >>>         self.seed = seed
        >>>         self.ingredient = ingredient
        >>> config = OmegaConf.create({
        >>>     '_component_': 'Food',
        >>>     'seed': 0,
        >>>     'ingredient': {'_component_': 'Spice', 'heat_level': 5}
        >>> })
        >>> food = instantiate(config, seed=42)
        >>> print(food.seed)  # 42
        >>> print(food.ingredient.heat_level)  # 5
        >>> new_spice = {'_component_': 'Spice', 'heat_level': 10}
        >>> food = instantiate(config, ingredient=new_spice)
        >>> print(food.ingredient.heat_level)  # 10

    Raises:
        ValueError: If config is not a DictConfig.

    Note: Modifies sys.path to include the current working directory for local imports.
    """
    if config is None:
        return None
    if not OmegaConf.is_dict(config):
        raise ValueError(f"instantiate only supports DictConfigs, got {type(config)}")

    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    config_copy = copy.deepcopy(config)
    config_copy._set_flag(
        flags=["allow_objects", "struct", "readonly"], values=[True, False, False]
    )
    config_copy._set_parent(config._get_parent())
    config = config_copy

    if kwargs:
        config = OmegaConf.merge(config, kwargs)

    OmegaConf.resolve(config)
    return _instantiate_node(OmegaConf.to_container(config, resolve=True), *args)
