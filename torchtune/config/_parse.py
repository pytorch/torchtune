# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import functools
import sys
from argparse import Namespace
from typing import Any, Callable, List, Tuple

from omegaconf import DictConfig, OmegaConf

from torchtune.config._utils import _merge_yaml_and_cli_args


Recipe = Callable[[DictConfig], Any]


class TuneRecipeArgumentParser(argparse.ArgumentParser):
    """
    A helpful utility subclass of the ``argparse.ArgumentParser`` that
    adds a builtin argument "config". The config argument takes a file path to a YAML file
    and loads in argument defaults from said file. The YAML file must only contain
    argument names and their values and nothing more, it does not have to include all of the
    arguments. These values will be treated as defaults and can still be overridden from the
    command line. Everything else works the same as the base ArgumentParser and you should
    consult the docs for more info: https://docs.python.org/3/library/argparse.html.

    Note:
        This class uses "config" as a builtin argument so it is not available to use.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        super().add_argument(
            "--config",
            type=str,
            help="Path/name of a yaml file with recipe args",
            required=True,
        )

    def parse_known_args(self, *args, **kwargs) -> Tuple[Namespace, List[str]]:
        """This acts the same as the base parse_known_args but will first load in defaults from
        from the config yaml file if it is provided. The command line args will always take
        precident over the values in the config file. All other parsing method, such as parse_args,
        internally call this method so they will inherit this property too. For more info see
        the docs for the base method: https://docs.python.org/3/library/argparse.html#the-parse-args-method.
        """
        namespace, unknown_args = super().parse_known_args(*args, **kwargs)

        unknown_flag_args = [arg for arg in unknown_args if arg.startswith("--")]
        if unknown_flag_args:
            raise ValueError(
                f"Additional flag arguments not supported: {unknown_flag_args}. Please use --config or key=value overrides"
            )

        config = OmegaConf.load(namespace.config)
        assert "config" not in config, "Cannot use 'config' within a config file"
        self.set_defaults(**config)

        namespace, unknown_args = super().parse_known_args(*args, **kwargs)
        del namespace.config

        return namespace, unknown_args


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
