# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import textwrap

from torchtune._cli.subcommand import Subcommand

from torchtune._recipe_registry import get_all_recipes


class List(Subcommand):
    """Holds all the logic for the `tune ls` subcommand."""

    NULL_VALUE = "<>"

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self._parser = subparsers.add_parser(
            "ls",
            prog="tune ls",
            help="List all built-in recipes and configs",
            description="List all built-in recipes and configs",
            epilog=textwrap.dedent(
                """\
            examples:
                $ tune ls
                RECIPE                                   CONFIG
                full_finetune_single_device              llama2/7B_full_single_device
                full_finetune_distributed                llama2/7B_full
                                                         llama2/13B_full
                ...

            To run one of these recipes:
                $ tune run full_finetune_single_device --config full_finetune_single_device
            """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._parser.add_argument(
            "--experimental",
            action="store_true",
            help="Includes experimental recipes and configs in output",
        )
        self._parser.set_defaults(func=self._ls_cmd)

    def _ls_cmd(self, args: argparse.Namespace) -> None:
        """List all available recipes and configs."""
        # Print table header
        header = f"{'RECIPE':<40} {'CONFIG':<40}"
        print(header)

        # Print recipe/config pairs
        for recipe in get_all_recipes(include_experimental=args.experimental):
            # If there are no configs for a recipe, print a blank config
            recipe_str = recipe.name
            if len(recipe.configs) == 0:
                row = f"{recipe_str:<40} {self.NULL_VALUE:<40}"
                print(row)
            for i, config in enumerate(recipe.configs):
                # If there are multiple configs for a single recipe, omit the recipe name
                # on latter configs
                if i > 0:
                    recipe_str = ""
                row = f"{recipe_str:<40} {config.name:<40}"
                print(row)
