# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import inspect
from typing import Any, Callable, Dict


def validate_recipe_args(recipe_function: Callable, args: Dict[str, Any]) -> None:
    """Ensure config and CLI args match with recipe and have the expected type

    Args:
        recipe_function (Callable): main method for recipe script, used to pull the expected
            parameters and their types
        args (Dict[str, Any]): dictionary of args passed to recipe

    Raises:
        TypeError: if args do not match the expected types
    """
    # Get params and type annotations
    sig = inspect.signature(recipe_function)
    params = sig.parameters
    for name, param in params.items():
        if name not in args:
            raise ValueError(f"Missing required argument {name}")
        arg = args.get(name)
        if not isinstance(arg, param.annotation):
            raise TypeError(
                f"Parameter {name} must be of type {param.annotation}, got {type(arg)}"
            )
    if len(args) > len(params):
        raise ValueError(f"Unknown arguments: {args.keys() - params.keys()}")
