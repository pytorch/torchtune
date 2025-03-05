# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import textwrap

from pathlib import Path
from typing import List, Optional

import yaml
from torchtune._cli.subcommand import Subcommand
from torchtune._recipe_registry import Config, get_all_recipes

ROOT = Path(__file__).parent.parent.parent


class Cat(Subcommand):
    """Holds all the logic for the `tune cat` subcommand."""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self._parser = subparsers.add_parser(
            "cat",
            prog="tune cat",
            help="Pretty print a config, making it easy to know which parameters you can override with `tune run`.",
            description="Pretty print a config, making it easy to know which parameters you can override with `tune run`.",
            epilog=textwrap.dedent(
                """\
                examples:
                    $ tune cat llama2/7B_full
                    output_dir: /tmp/torchtune/llama2_7B/full
                    tokenizer:
                        _component_: torchtune.models.llama2.llama2_tokenizer
                        path: /tmp/Llama-2-7b-hf/tokenizer.model
                        max_seq_len: null
                    ...

                    # Pretty print the config in sorted order
                    $ tune cat llama2/7B_full --sort

                    # Pretty print the contents of LOCALFILE.yaml
                    $ tune cat LOCALFILE.yaml

                You can now easily override a key based on your findings from `tune cat`:
                    $ tune run full_finetune_distributed --config llama2/7B_full output_dir=./

                Need to find all the "cat"-able configs? Try `tune ls`!
                """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._parser.add_argument(
            "config_name", type=str, help="Name of the config to print"
        )
        self._parser.set_defaults(func=self._cat_cmd)
        self._parser.add_argument(
            "--sort", action="store_true", help="Print the config in sorted order"
        )

    def _get_all_recipes(self) -> List[str]:
        return [recipe.name for recipe in get_all_recipes()]

    def _get_config(self, config_str: str) -> Optional[Config]:
        # Search through all recipes
        for recipe in get_all_recipes():
            for config in recipe.configs:
                if config.name == config_str:
                    return config

    def _print_yaml_file(self, file: str, sort_keys: bool) -> None:
        try:
            with open(file, "r") as f:
                data = yaml.safe_load(f)
                if data:
                    print(
                        yaml.dump(
                            data,
                            default_flow_style=False,
                            sort_keys=sort_keys,
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

        self._print_yaml_file(str(config_path), args.sort)
