# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import runpy
import sys
import textwrap

from pathlib import Path
from typing import Optional

import torchtune

from torch.distributed.run import get_args_parser as get_torchrun_args_parser, run
from torchtune._cli.subcommand import Subcommand
from torchtune._recipe_registry import Config, get_all_recipes, Recipe

ROOT = Path(torchtune.__file__).parent.parent


class Run(Subcommand):
    """Holds all the logic for the `tune run` subcommand."""

    def __init__(self, subparsers):
        super().__init__()
        self._parser = subparsers.add_parser(
            "run",
            prog="tune run",
            help="Run a recipe. For distributed recipes, this supports all torchrun arguments.",
            description="Run a recipe. For distributed recipes, this supports all torchrun arguments.",
            usage="tune run [TORCHRUN-OPTIONS] <recipe> --config <config> [RECIPE-OPTIONS]",
            epilog=textwrap.dedent(
                """\
                examples:

                    # Run a finetuning recipe on a single device w/ default values
                    $ tune run lora_finetune_single_device --config llama2/7B_lora_single_device

                    # Run a finetuning recipe in a distributed fashion using torchrun w/ default values
                    $ tune run --nproc_per_node 4 full_finetune_distributed --config llama2/7B_full_finetune_distributed

                    # Override a parameter in the config file and specify a number of GPUs for torchrun
                    $ tune run --nproc_per_node 2 \
                        lora_finetune_single_device \
                        --config llama2/7B_lora_single_device \
                        model.lora_rank=16 \

                Remember, you can use `tune cp` to copy a default recipe/config to your local dir and modify the values.
                """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self._parser.set_defaults(func=self._run_cmd)

    def _add_arguments(self) -> None:
        """Add arguments to the parser.

        This is a bit hacky since we need to add the torchrun arguments to our parser.
        This grabs the argparser from torchrun, iterates over it's actions, and adds them
        to our parser. We rename the training_script and training_script_args to recipe and recipe_args
        respectively. In addition, we leave out the help argument since we add it manually to ours.
        """
        torchrun_argparser = get_torchrun_args_parser()
        for action in torchrun_argparser._actions:
            if action.dest == "training_script":
                action.dest = "recipe"
                action.help = """Name or path to recipe to be launched followed by args.
For a list of all possible recipes, run `tune ls`."""
            elif action.dest == "training_script_args":
                action.dest = "recipe_args"
                action.help = "Args to be passed to the recipe."
            elif action.dest == "help":
                continue
            self._parser._add_action(action)

    def _run_distributed(self, args: argparse.Namespace):
        """Run a recipe with torchrun."""
        # TODO (rohan-varma): Add check that nproc_per_node <= cuda device count. Currently,
        # we don't do this since we test on CPUs for distributed. Will update once multi GPU CI is supported.
        print("Running with torchrun...")
        # Have to reset the argv so that the recipe can be run with the correct arguments
        args.training_script = args.recipe
        args.training_script_args = args.recipe_args
        run(args)

    def _run_single_device(self, args: argparse.Namespace):
        """Run a recipe on a single device."""
        sys.argv = [str(args.recipe)] + args.recipe_args
        runpy.run_path(str(args.recipe), run_name="__main__")

    def _is_distributed_args(self, args: argparse.Namespace):
        """Check if the user is trying to run a distributed recipe."""
        total = len(sys.argv) - 2  # total args minus "tune run"
        script_args = len(args.recipe_args) + 1  # script args + 1 for script name
        return total > script_args

    def _get_recipe(self, recipe_str: str) -> Optional[Recipe]:
        """Get a recipe from the name or path.

        Args:
            recipe_str (str): The name or path of the recipe.

        Returns:
            The recipe if it's found in built-in recipes, otherwise None.
        """
        for recipe in get_all_recipes():
            if recipe.name == recipe_str:
                return recipe

    def _get_config(
        self, config_str: str, specific_recipe: Optional[Recipe]
    ) -> Optional[Config]:
        """Get a config from the name or path.

        Args:
            config_str (str): The name or path of the config.
            specific_recipe (Optional[Recipe]): The specific recipe to search through.

        Returns:
            The config if it's found in built-in configs, otherwise None.
        """
        # If a specific recipe is provided, search through it
        if specific_recipe is not None:
            for config in specific_recipe.configs:
                if config.name == config_str:
                    return config

        # If not, search through all recipes
        for recipe in get_all_recipes():
            for config in recipe.configs:
                if config.name == config_str:
                    return config

    def _run_cmd(self, args: argparse.Namespace):
        """Run a recipe."""
        # We have to assume that the recipe supports distributed training
        supports_distributed = True
        recipe_path, config_path = None, None

        # Try to find config string in args
        try:
            config_idx = args.recipe_args.index("--config") + 1
            config_str = args.recipe_args[config_idx]
        except ValueError:
            self._parser.error("The '--config' argument is required.")

        # Get recipe path
        recipe = self._get_recipe(args.recipe)
        if recipe is None:
            recipe_path = args.recipe
        else:
            recipe_path = str(ROOT / "recipes" / recipe.file_path)
            supports_distributed = recipe.supports_distributed

        # Get config path
        config = self._get_config(config_str, recipe)
        if config is None:
            config_path = config_str
        else:
            config_path = str(ROOT / "recipes" / "configs" / config.file_path)

        # Prepare args
        args.recipe = recipe_path
        args.recipe_args[config_idx] = config_path

        # Execute recipe
        if self._is_distributed_args(args):
            if not supports_distributed:
                self._parser.error(
                    f"Recipe {recipe.name} does not support distributed training."
                    "Please run without torchrun commands."
                )
            self._run_distributed(args)
        else:
            self._run_single_device(args)
