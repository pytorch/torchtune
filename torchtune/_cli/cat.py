# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import textwrap

from pathlib import Path
from typing import List, Optional

import torchtune
import yaml
from torchtune._cli.subcommand import Subcommand
from torchtune._recipe_registry import Config, get_all_recipes

ROOT = Path(torchtune.__file__).parent.parent


class Cat(Subcommand):
    """Holds all the logic for the `tune cat` subcommand."""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self._parser = subparsers.add_parser(
            "cat",
            prog="tune cat",
            help="Pretty print a config",
            description="Pretty print a config",
            epilog=textwrap.dedent(
                """\
                examples:
                    $ tune cat llama2/7B_full
                    model: llama2
                    size: 7B
                    task: full_finetune
                    ...

                    $ tune cat non_existent_config
                    Config 'non_existent_config' not found.

                    $ tune cat some_recipe
                    'some_recipe' is a recipe, not a config. Please use a config name.
                """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._parser.add_argument(
            "config_name", type=str, help="Name of the config to print"
        )
        self._parser.set_defaults(func=self._cat_cmd)

    def _get_all_recipes(self) -> List[str]:
        return [recipe.name for recipe in get_all_recipes()]

    def _get_config(self, config_str: str) -> Optional[Config]:
        # Search through all recipes
        for recipe in get_all_recipes():
            for config in recipe.configs:
                if config.name == config_str:
                    return config

    def _print_file(self, file: str) -> None:
        try:
            with open(file, "r") as f:
                data = yaml.safe_load(f)
                if data:
                    print(
                        yaml.dump(
                            data,
                            default_flow_style=False,
                            sort_keys=False,
                            indent=4,
                            width=80,
                            allow_unicode=True,
                        ),
                        end="",
                    )
        except yaml.YAMLError as e:
            self._parser.error(f"Error parsing YAML file: {e}")

    def _cat_cmd(self, args: argparse.Namespace) -> None:
        """Display the contents of a configuration file.

        Handles both predefined configurations and direct file paths, ensuring:
        - Input is not a recipe name
        - File exists
        - File is YAML format

        Args:
            args (argparse.Namespace): Command-line arguments containing 'config_name' attribute
        """
        config_str = args.config_name

        # Immediately handle recipe name case
        if config_str in self._get_all_recipes():
            print(
                f"'{config_str}' is a recipe, not a config. Please use a config name."
            )
            return

        # Resolve config path
        config = self._get_config(config_str)
        if config:
            config_path = ROOT / "recipes" / "configs" / config.file_path
        else:
            config_path = Path(config_str)
            if config_path.suffix.lower() not in {".yaml", ".yml"}:
                self._parser.error(
                    f"Invalid config format: '{config_path}'. Must be YAML (.yaml/.yml)"
                )
                return

        if not config_path.exists():
            self._parser.error(f"Config '{config_str}' not found.")
            return

        self._print_file(str(config_path))
