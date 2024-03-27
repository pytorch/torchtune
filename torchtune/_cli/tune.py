# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import os
import runpy
import shutil
import sys
import textwrap
from pathlib import Path

import torchtune

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from torch.distributed.run import get_args_parser as get_torchrun_args_parser, run

from torchtune import config as config_mod, get_all_recipes
from torchtune.config._errors import ConfigError

ROOT = Path(torchtune.__file__).parent.parent.absolute()


class Subcommand:
    """Base class for all subcommands."""

    def __init__(self, *args, **kwargs):
        self._parser = None

    @classmethod
    def create(cls, *args, **kwargs):
        """Create a new instance of the subcommand."""
        return cls(*args, **kwargs)


class List(Subcommand):
    """Holds all the logic for the `tune ls` subcommand."""

    NULL_VALUE = "<>"

    def __init__(self, subparsers):
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
        self._parser.set_defaults(func=self._ls_cmd)

    def _ls_cmd(self, args):
        """List all available recipes and configs."""
        # Print table header
        header = f"{'RECIPE':<40} {'CONFIG':<40}"
        print(header)

        # Print recipe/config pairs
        for recipe in get_all_recipes():
            # If there are no configs for a recipe, print a blank config
            recipe_str = recipe.name
            if len(recipe.get_configs()) == 0:
                row = f"{recipe_str:<40} {self.NULL_VALUE:<40}"
                print(row)
            for i, config in enumerate(recipe.get_configs()):
                # If there are multiple configs for a single recipe, omit the recipe name
                # on latter configs
                if i > 0:
                    recipe_str = ""
                row = f"{recipe_str:<40} {config.name:<40}"
                print(row)


class Copy(Subcommand):
    """Holds all the logic for the `tune cp` subcommand."""

    def __init__(self, subparsers):
        super().__init__()
        self._parser = subparsers.add_parser(
            "cp",
            prog="tune cp",
            usage="tune cp <recipe|config> destination [OPTIONS]",
            help="Copy a built-in recipe or config to a local path.",
            description="Copy a built-in recipe or config to a local path.",
            epilog=textwrap.dedent(
                """\
            examples:
                $ tune cp lora_finetune_distributed .
                Copied file to ./lora_finetune_distributed.py

                $ tune cp llama2/7B_full ./new_dir/my_custom_lora.yaml --make-parents
                Copyied file to ./new_dir/my_custom_lora.yaml

            Need to see all possible recipes/configs to copy? Try running `tune ls`.
            """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._parser.add_argument(
            "file",
            type=str,
            help="Recipe/config to copy. For a list of all possible options, run `tune ls`",
        )
        self._parser.add_argument(
            "destination",
            type=Path,
            help="Location to copy the file to",
        )
        self._parser.add_argument(
            "-n",
            "--no-clobber",
            action="store_true",
            help="Do not overwrite destination if it already exists",
            default=False,
        )
        self._parser.add_argument(
            "--make-parents",
            action="store_true",
            help="Create parent directories for destination if they do not exist. "
            "If not set to True, will error if parent directories do not exist",
            default=False,
        )
        self._parser.set_defaults(func=self._cp_cmd)

    def _cp_cmd(self, args: argparse.Namespace):
        """Copy a recipe or config to a new location."""
        destination: Path = args.destination
        src = None

        # Iterate through all recipes and configs
        for recipe in get_all_recipes():
            if recipe.name == args.file:
                src = ROOT / "recipes" / recipe.file_path
                proper_suffix = ".py"
                break
            for config in recipe.configs:
                if config.name == args.file:
                    src = ROOT / "recipes" / "configs" / config.file_path
                    proper_suffix = ".yaml"
                    break

        # Fail if no file exists
        if src is None:
            self._parser.error(
                f"Invalid file name: {args.file}. Try `tune ls` to see all available files to copy."
            )

        # Attach proper suffix if needed
        if destination != "." and destination.suffix != proper_suffix:
            destination = destination.with_suffix(proper_suffix)

        # Copy file
        try:
            if args.no_clobber and destination.exists():
                print(
                    f"File already exists at {destination.absolute()}, not overwriting."
                )
            else:
                if args.make_parents:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                output = shutil.copy(src, destination)
                print(f"Copied file to {output}")
        except FileNotFoundError:
            self._parser.error(
                f"Cannot create regular file: '{destination}'. No such file or directory. "
                "If the specified destination's parent directory does not exist and you would "
                "like to create it on-the-fly, use the --make-parents flag."
            )


class Download(Subcommand):
    """Holds all the logic for the `tune download` subcommand."""

    def __init__(self, subparsers):
        super().__init__()
        self._parser = subparsers.add_parser(
            "download",
            prog="tune download",
            usage="tune download <repo-id> [OPTIONS]",
            help="Download a model from the HuggingFace Hub.",
            description="Download a model from the HuggingFace Hub.",
            epilog=textwrap.dedent(
                """\
            examples:
                # Download a model from the HuggingFace Hub with a Hugging Face API token
                $ tune download meta-llama/Llama-2-7b-hf --hf-token <TOKEN> --output-dir /tmp/model
                Succesfully downloaded model repo and wrote to the following locations:
                /tmp/model/config.json
                /tmp/model/README.md
                /tmp/model/consolidated.00.pth
                ...

                # Download an ungated model from the HuggingFace Hub
                $ tune download mistralai/Mistral-7B-Instruct-v0.2
                Succesfully downloaded model repo and wrote to the following locations:
                ./model/config.json
                ./model/README.md
                ./tmp/model/model-00003-of-00003.safetensors
                ...

            For a list of all downloadable models, visit the HuggingFace Hub https://huggingface.co/models.
            """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._parser.add_argument(
            "repo_id",
            type=str,
            help="Name of the repository on HuggingFace Hub.",
        )
        self._parser.add_argument(
            "--output-dir",
            type=Path,
            required=False,
            default="./model",
            help="Directory in which to save the model.",
        )
        self._parser.add_argument(
            "--hf-token",
            type=str,
            required=False,
            default=os.getenv("HF_TOKEN", None),
            help="Hugging Face API token. Needed for gated models like Llama2.",
        )
        self._parser.set_defaults(func=self._download_cmd)

    def _download_cmd(self, args: argparse.Namespace) -> None:
        """Downloads a model from the Hugging Face Hub."""
        if "meta-llama" in args.repo_id and args.hf_token is None:
            self._parser.error(
                "You need to provide a Hugging Face API token to download gated models."
                "You can find your token by visiting https://huggingface.co/settings/tokens"
            )

        # Download the tokenizer and PyTorch model files
        try:
            true_output_dir = snapshot_download(
                args.repo_id,
                local_dir=args.output_dir,
                resume_download=True,
                token=args.hf_token,
            )
        except Exception as e:
            self._parser.error(str(e))

        print(
            "Succesfully downloaded model repo and wrote to the following locations:",
            *list(Path(true_output_dir).iterdir()),
            sep="\n",
        )


class Validate(Subcommand):
    """Holds all the logic for the `tune validate` subcommand."""

    def __init__(self, subparsers):
        super().__init__()
        self._parser = subparsers.add_parser(
            "validate",
            prog="tune validate",
            help="Validate a config and ensure that it is well-formed.",
            description="Validate a config and ensure that it is well-formed.",
            usage="tune validate <config>",
            epilog=textwrap.dedent(
                """\
                examples:

                    $ tune validate recipes/configs/full_finetune_distributed.yaml
                    Config is well-formed!

                """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._parser.add_argument(
            "config",
            type=Path,
            help="Path to the config to validate.",
        )
        self._parser.set_defaults(func=self._validate_cmd)

    def _validate_cmd(self, args: argparse.Namespace):
        """Validate a config file."""
        cfg = OmegaConf.load(args.config)

        try:
            config_mod.validate(cfg)
        except ConfigError as e:
            self._parser.error(str(e))

        print("Config is well-formed!")


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
                    $ tune run --num-gpu=4 full_finetune_distributed --config llama2/7B_full_finetune_distributed

                    # Override a parameter in the config file and specify a number of GPUs for torchrun
                    $ tune run lora_finetune_single_device \
                        --config llama2/7B_lora_single_device \
                        model.lora_rank=16 \

                Remember, you can use `tune cp` to copy a default recipe/config to your local dir and modify the values.
                """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        # Apply torchrun args to run parser
        torchrun_argparser = get_torchrun_args_parser()
        for action in torchrun_argparser._actions:
            if action.dest == "nproc_per_node":
                action.option_strings += ["--num-gpu", "--num_gpu"]
                action.help += "If `num_gpu` used, max number is 8 - does not support multi-node training."
            elif action.dest == "training_script":
                action.dest = "recipe"
                action.help = """
                    Name or path to recipe to be launched followed by args.
                    For a list of all possible recipes, run `tune ls`."""
            elif action.dest == "training_script_args":
                action.dest = "recipe_args"
                action.help = "Args to be passed to the recipe."
            elif action.dest == "help":
                continue
            self._parser._add_action(action)
        self._parser.set_defaults(func=self._run_cmd)

    def _run_distributed(self, args: argparse.Namespace):
        """Run a recipe with torchrun."""
        # TODO (rohan-varma): Add check that nproc_per_node <= cuda device count. Currently,
        # we don't do this since we test on CPUs for distributed. Will update once multi GPU CI is supported.
        print("Running with torchrun...")
        # Have to reset the argv so that the recipe can be run with the correct arguments
        args = copy.deepcopy(args)
        args.__dict__["training_script"] = args.__dict__.pop("recipe")
        args.__dict__["training_script_args"] = args.__dict__.pop("recipe_args")
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

    def _run_cmd(self, args: argparse.Namespace):
        """Run a recipe."""
        config_idx = args.recipe_args.index("--config") + 1
        config_str = args.recipe_args[config_idx]

        for recipe in get_all_recipes():
            if recipe.name == args.recipe:
                # Locate the actual file
                full_recipe_path = str(ROOT / "recipes" / recipe.file_path)
                args.recipe = full_recipe_path

                # Attempt to locate the config file
                for config in recipe.get_configs():
                    if config_str == config.name:
                        full_config_path = str(
                            ROOT / "recipes" / "configs" / config.file_path
                        )
                        args.recipe_args[config_idx] = full_config_path

                # Handle distributed vs. non-distributed cases
                if self._is_distributed_args(args):
                    if not recipe.supports_distributed:
                        self._parser.error(
                            f"Recipe {recipe.name} does not support distributed training. Please run without torchrun commands."
                        )
                    else:
                        self._run_distributed(args)
                        return
                else:
                    self._run_single_device(args)
                    return

        # Handle the case where the recipe name is not known and the config is likely a default config
        if not config_str.endswith(".yaml"):
            self._parser.error(
                "It looks like you might be trying to use a default config with a custom recipe script. "
                "This is not currently supported, please copy the config file to your local dir first with "
                f"`tune cp {config_str} .` Then specify the new config file in your `tune run ...` command."
            )

        # Handle the case where we don't know the recipe name
        if self._is_distributed_args(args):
            self._run_distributed(args)
            return
        else:
            self._run_single_device(args)
            return


class TuneCLIParser:
    """Holds all information related to running the CLI"""

    def __init__(self):
        # Initialize the top-level parser
        self._parser = argparse.ArgumentParser(
            prog="tune",
            description="Welcome to the TorchTune CLI!",
            add_help=True,
        )
        self._parser.set_defaults(func=lambda args: self._parser.print_help())

        # Add subcommands
        subparsers = self._parser.add_subparsers(title="subcommands")
        Download.create(subparsers)
        List.create(subparsers)
        Copy.create(subparsers)
        Run.create(subparsers)
        Validate.create(subparsers)

    def parse_args(self) -> argparse.Namespace:
        """Parse CLI arguments"""
        return self._parser.parse_args()

    def run(self, args: argparse.Namespace):
        """Execute CLI"""
        args.func(args)


def main():
    """Entrypoint for CLI"""
    parser = TuneCLIParser()
    args = parser.parse_args()
    parser.run(args)


if __name__ == "__main__":
    main()
