# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import functools
import sys
from typing import Any, Callable

from omegaconf import DictConfig
from torchtune.config._utils import _merge_yaml_and_cli_args
from torchtune.utils.argparse import TuneRecipeArgumentParser


Recipe = Callable[[DictConfig], Any]


def parse(recipe_main: Recipe) -> Callable[[Recipe], Any]:
    """
    Decorator that handles parsing the config file and CLI overrides
    for a recipe. Use it on the recipe's main function.

    Args:
        recipe_main (Recipe): The main method that initializes
            and runs the recipe

    Examples:
        >>> @parse
        >>> def main(cfg: DictConfig):
        >>>     ...

        >>> # With the decorator, the parameters will be parsed into cfg when run as:
        >>> tune my_recipe --config config.yaml foo=bar

    Returns:
        Callable[[Recipe], Any]: the decorated main
    """

    @functools.wraps(recipe_main)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        parser = TuneRecipeArgumentParser(
            description=recipe_main.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        # Get user-specified args from config and CLI and create params for recipe
        yaml_args, cli_args = parser.parse_known_args()
        conf = _merge_yaml_and_cli_args(yaml_args, cli_args)

        sys.exit(recipe_main(conf))

    return wrapper
