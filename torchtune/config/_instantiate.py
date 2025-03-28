# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy

import inspect
import os
import sys
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf
from torchtune.config._errors import InstantiationError
from torchtune.config._utils import _get_component_from_path


def _create_component(
    _component_: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """
    Create an instance of a class or a partial function with given arguments.

    If the component is a class, instantiate it with the provided positional and keyword arguments.
    If the component is a callable (e.g., a function), return a partial function with the arguments
    pre-applied using functools.partial.

    Args:
        _component_ (Callable[..., Any]): The component (class or function) to process.
        args (Tuple[Any, ...]): Positional arguments for the component.
        kwargs (Dict[str, Any]): Keyword arguments for the component.

    Returns:
        Any: An instance of the class or a partial function.

    Raises:
        InstantiationError: If the component is not callable.
    """
    if inspect.isclass(_component_):
        return _component_(*args, **kwargs)
    elif callable(_component_):
        if args or kwargs:
            return partial(_component_, *args, **kwargs)
        else:
            return _component_
    else:
        raise InstantiationError(f"Cannot process non-callable {_component_}")


def _instantiate_node(
    obj: Any,
    *args: Any,
    caller_globals: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create an instance or a partial function from an object, recursively processing nested structures.

    If the object is a dictionary with a '_component_' key, create the specified component (class instance
    or partial function), processing its arguments recursively. If it's a dictionary without '_component_'
    or a list, process each item recursively. Otherwise, return the object unchanged.

    Args:
        obj (Any): Object to process (dict, list, or other).
        *args (Any): Positional arguments for the component (used only at top level).
        caller_globals (Optional[Dict[str, Any]]): Enable instantiating objects from caller's globals.

    Returns:
        Any: Instantiated object, partial function, or processed structure.
    """
    if isinstance(obj, dict) or isinstance(obj, DictConfig):
        if "_component_" not in obj:
            return {
                k: _instantiate_node(v, caller_globals=caller_globals)
                for k, v in obj.items()
            }
        else:
            _component_ = _get_component_from_path(
                obj["_component_"], caller_globals=caller_globals
            )
            kwargs = {
                k: _instantiate_node(v, caller_globals=caller_globals)
                for k, v in obj.items()
                if k != "_component_"
            }
            return _create_component(_component_, args, kwargs)
    elif isinstance(obj, list):
        return [_instantiate_node(item, caller_globals=caller_globals) for item in obj]
    else:
        return obj


def instantiate(
    config: Union[Dict[str, Any], DictConfig],
    *args: Any,
    caller_globals: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """
    Create an instance of a class or a partial function from a configuration, recursively handling nested components.

    Given a dict with a '_component_' field specifying the class or callable to use and additional fields
    as arguments, create an instance if it's a class or a partial function if it's a callable (e.g., a function).
    Positional and keyword arguments passed in the call are merged with the config, with keyword arguments
    taking precedence.

    Based on Hydra's `instantiate` utility.

    Args:
        config (Union[Dict[str, Any], DictConfig]): Configuration with '_component_' and optional arguments.
        *args (Any): Positional arguments for the component.
        caller_globals (Optional[Dict[str, Any]]): Enable instantiating objects from caller's globals.
        **kwargs (Any): Keyword arguments to override or add to the config.

    Returns:
        Any: The created object (class instance or partial function), or None if config is None.

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
        >>>
        >>> # Example with a function
        >>> config = {'_component_': 'torch.nn.functional.cross_entropy', 'ignore_index': 999}
        >>> loss = instantiate(config)
        >>> # loss is a partial function: partial(cross_entropy, ignore_index=999)
        >>> # Call it with remaining arguments:
        >>> input = ...  # torch tensor
        >>> target = ...  # torch tensor
        >>> result = loss(input, target)  # Computes cross_entropy(input, target, ignore_index=999)

    Raises:
        ValueError: If config is not a DictConfig or dict.
        InstantiationError: If the object to instantiate misses the '_component_' key or cannot be resolved.
    """
    if config is None:
        return None

    # Convert plain dict to DictConfig if necessary
    if isinstance(config, dict):
        config = OmegaConf.create(config)

    elif not OmegaConf.is_dict(config):
        raise ValueError(
            f"instantiate only supports DictConfigs or dicts, got {type(config)}"
        )

    if "_component_" not in config:
        raise InstantiationError(
            "Cannot instantiate specified object."
            + "\nMake sure you've specified a _component_ field with a valid dotpath."
            + f"\nGot {config=}."
        )

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

    # caller → instantiate → _instantiate_node → _get_component_from_path
    # To get the caller's globals, in case the the user is trying to instantiate some object from it,
    # we step back (f_back) and get it, so `_get_component_from_path` can use it.
    if caller_globals is None:
        current_frame = inspect.currentframe()
        if current_frame and current_frame.f_back:
            caller_globals = current_frame.f_back.f_globals

    return _instantiate_node(
        OmegaConf.to_container(config, resolve=True),
        caller_globals=caller_globals,
        *args,
    )
