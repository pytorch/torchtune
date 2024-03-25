# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO (rohan-varma): Add check that nproc_per_node <= cuda device count. Currently,
# we don't do this since we test on CPUs for distributed. Will update once multi GPU
# CI is supported.

import argparse
import os
import shutil
import textwrap

from pathlib import Path

import torch

import torchtune

from huggingface_hub import snapshot_download

from torch.distributed.run import (
    get_args_parser as get_torchrun_args_parser,
    run as _torchrun_cmd,
)

from torchtune import config, list_configs, list_recipes
from torchtune.config._utils import _merge_yaml_and_cli_args

from torchtune.models.llama2 import convert_llama2_fair_format
from torchtune.utils import get_logger
from torchtune.utils.constants import MODEL_KEY

ROOT = Path(torchtune.__file__).parent.parent.absolute()
NULL_VALUE = "<>"
PYTORCH_MODEL_FILENAME = "native_pytorch_model.pt"


class TuneArgumentParser:
    """Holds all information related to running the CLI"""

    def __init__(self):
        self._logger = get_logger("INFO")
        self._parser = argparse.ArgumentParser(
            prog="tune",
            description="Welcome to the TorchTune CLI!",
            add_help=True,
            exit_on_error=True,
        )
        self._parser.set_defaults(func=lambda args: tune_parser.print_help())
        subparsers = self._parser.add_subparsers(title="subcommands")

        # Add `ls` command
        ls_parser = subparsers.add_parser(
            "ls",
            prog="tune ls",
            help="List all built-in recipes and configs",
            epilog=textwrap.dedent(
                """\
            examples:
                $ tune ls
                RECIPE                           CONFIG
                full_finetune_distributed.py     full_finetune_distributed.yaml
                lora_finetune_distributed.py     lora_finetune_distributed.yaml
                alpaca_generate.py               alpaca_generate.yaml

            To run one of these recipes:
                $ tune full_finetune_single_device --config full_finetune_single_device
            """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        ls_parser.set_defaults(func=self._ls_cmd)

        # Add `cp` command
        cp_parser = subparsers.add_parser(
            "cp",
            prog="tune cp",
            usage="tune cp <recipe|config> destination [OPTIONS]",
            help="Copy a built-in recipe or config to a local path.",
            epilog=textwrap.dedent(
                """\
            examples:
                $ tune cp lora_finetune_distributed.yaml ./my_custom_llama2_lora.yaml
                $ tune cp full_finetune_distributed.py ./my_custom_full_finetune.py
                $ tune cp full_finetune_distributed.py ./new_dir/my_custom_full_finetune.py --make-parents

            Need to see all possible recipes/configs to copy? Try running `tune ls`.
            And as always, you can also run `tune cp --help` for more information.
            """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        cp_parser.add_argument(
            "file",
            type=str,
            help="Recipe/config to copy. For a list of all possible options, run `tune ls`",
        )
        cp_parser.add_argument(
            "destination",
            type=Path,
            help="Location to copy the file to",
        )
        cp_parser.add_argument(
            "-n",
            "--no-clobber",
            action="store_true",
            help="Do not overwrite destination if it already exists",
            default=False,
        )
        cp_parser.add_argument(
            "--make-parents",
            action="store_true",
            help="Create parent directories for destination if they do not exist. "
            "If not set to True, will error if parent directories do not exist",
            default=False,
        )
        cp_parser.set_defaults(func=self._cp_cmd)

        # Add `download` command
        download_parser = subparsers.add_parser(
            "download",
            prog="tune download",
            usage="tune download <repo-id> [OPTIONS]",
            help="Download a model from the Hugging Face Hub.",
        )
        download_parser.add_argument(
            "repo_id",
            type=str,
            help="Name of the repository on Hugging Face Hub.",
        )
        download_parser.add_argument(
            "--output-dir",
            type=Path,
            required=False,
            default="/tmp/model",
            help="Directory in which to save the model.",
        )
        download_parser.add_argument(
            "--hf-token",
            type=str,
            required=False,
            default=os.getenv("HF_TOKEN", None),
            help="Hugging Face API token. Needed for gated models like Llama2.",
        )
        download_parser.set_defaults(func=self._download_cmd)

        # Add `convert_checkpoint` command
        convert_ckpt_parser = subparsers.add_parser(
            "convert_checkpoint",
            prog="tune convert_checkpoint",
            usage="tune convert_checkpoint <ckpt-path> --output-path <path> --model <model> --train-type <type> [OPTIONS]",
            help="Convert a model checkpoint to a format compatible with TorchTune.",
        )
        convert_ckpt_parser.add_argument(
            "checkpoint-path", type=Path, help="Path to the checkpoint to convert."
        )
        convert_ckpt_parser.add_argument(
            "--output-path",
            type=Path,
            help="Where to write the converted checkpoint. "
            "Will default to the same directory as the original checkpoint if no arg is provided"
            f"under the filename {PYTORCH_MODEL_FILENAME}.",
            required=False,
            default=None,
        )
        convert_ckpt_parser.add_argument(
            "--model",
            type=str,
            help="model name",
            choices=["llama2"],
            required=True,
        )
        convert_ckpt_parser.add_argument(
            "--train-type",
            type=str,
            help="Type of finetuning. Currently Full-Finetuning and LoRA have slightly different formats. "
            "This will be resolved soon.",
            choices=["full", "lora"],
            required=True,
        )
        convert_ckpt_parser.add_argument(
            "--output-numerical-validation",
            action="store_true",
            help="Whether to load the original checkpoint and the converted checkpoint and compare"
            "the numerical output of a forward pass to ensure that the conversion was successful."
            "Prints results to stdout. This additional check is only available for Llama2 7B."
            "This will take awhile and may consume lots of memory. If you see an OOM error,"
            "please disable this flag. Note: All our checkpoints conversions are already validated"
            "in unit tests for smaller checkpoints and integration tests for larger checkpoints."
            "This flag is primarily for debugging purposes.",
            required=False,
            default=False,
        )
        convert_ckpt_parser.set_defaults(func=self._convert_checkpoint_cmd)

        # Add `validate` command
        validate_parser = subparsers.add_parser(
            "validate",
            prog="tune validate",
            help="Validate a config and ensure that it is well-formed.",
            usage="tune validate --config recipes/configs/full_finetune_distributed.yaml",
            epilog=textwrap.dedent(
                """\
                examples:

                    $ tune validate --config recipes/configs/full_finetune_distributed.yaml
                    Config is well-formed!

                """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        validate_parser.set_defaults(func=self._validate_cmd)

        # Add `run` command
        run_parser = subparsers.add_parser(
            "run",
            prog="tune run",
            help="Run a recipe. This is a wrapper around torchrun so you can also use any torchrun args.",
            usage="tune run <recipe> --config <config> [RECIPE-OPTIONS] [TORCHRUN-OPTIONS]",
            epilog=textwrap.dedent(
                """\
                examples:

                    $ tune run full_finetune_distributed.py \
                        --config recipes/configs/full_finetune_distributed.yaml \
                        --num-gpu 4 \

                    # Override a parameter in the config file and specify a number of GPUs for torchrun
                    $ tune run lora_finetune_distributed.py \
                        --config recipes/configs/lora_finetune_distributed.yaml \
                        model.lora_rank=16 \
                        --num-gpu 4 \
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
            run_parser._add_action(action)
        run_parser.set_defaults(func=_torchrun_cmd)

    def _cp_cmd(self, args: argparse.Namespace):
        """Copy a recipe or config to a new location."""
        destination = args.destination

        # Check if recipe/config is valid
        all_recipes_and_configs = list_recipes() + [
            config for recipe in list_recipes() for config in list_configs(recipe)
        ]
        if args.file not in all_recipes_and_configs:
            self._parser.error(
                f"Invalid file name: {args.file}. Try `tune ls` to see all available files to copy."
            )

        # Get file path
        file_name = args.file
        src = ROOT / "recipes" / file_name

        # Copy file
        try:
            if args.no_clobber and destination.exists():
                print(
                    f"File already exists at {destination.absolute()}, not overwriting."
                )
            else:
                if args.make_parents:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, destination)
        except FileNotFoundError:
            self._parser.error(
                f"Cannot create regular file: '{destination}'. No such file or directory. "
                "If the specified destination's parent directory does not exist and you would "
                "like to create it on-the-fly, use the --make-parents flag."
            )

    def _ls_cmd(self, args):
        """List all available recipes and configs."""
        # Print table header
        header = f"{'RECIPE':<40} {'CONFIG':<40}"
        print(header)

        # Print recipe/config pairs
        for recipe in list_recipes():
            configs = list_configs(recipe)
            # If there are no configs for a recipe, print a blank config
            if len(configs) == 0:
                row = f"{recipe:<40} {NULL_VALUE:<40}"
                print(row)
            for i, config_name in enumerate(configs):
                # If there are multiple configs for a single recipe, omit the recipe name
                # on latter configs
                if i > 0:
                    recipe = ""
                row = f"{recipe:<40} {config_name:<40}"
                print(row)

    def _download_cmd(self, args) -> None:
        """Downloads a model from the Hugging Face Hub."""
        if "meta-llama" in args.repo_id and args.hf_token is None:
            raise self._parser.error(
                "You need to provide a Hugging Face API token to download gated models."
                "You can find your token by visiting https://huggingface.co/settings/tokens"
            )

        # Download the tokenizer and PyTorch model files
        true_output_dir = snapshot_download(
            args.repo_id,
            local_dir=args.output_dir,
            resume_download=True,
            token=args.hf_token,
        )

        print(
            "Succesfully downloaded model repo and wrote to the following locations:",
            *list(Path(true_output_dir.iterdir())),
            sep="\n",
        )

    def _validate_cmd(self, args: argparse.Namespace):
        """Validate a config file."""
        yaml_args = args.yaml_args
        cli_args = args.cli_args
        conf = _merge_yaml_and_cli_args(yaml_args, cli_args)
        config.validate(cfg)
        print("Config is well-formed!")

    def _convert_checkpoint_cmd(self, args: argparse.Namespace):
        """Convert model checkpoint to a PyTorch-native format compatible with Torchtune."""
        checkpoint_path = args.checkpoint_path
        model = args.model
        output_path = args.output_path
        train_type = args.train_type
        output_numerical_validation = args.output_numerical_validation

        # Load the original state dict
        original_state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        self._logger.info(msg="Loaded original state dict")

        # Convert checkpoint
        if model == "llama2":
            state_dict = convert_llama2_fair_format(
                original_state_dict, output_numerical_validation
            )
        else:
            self._parser.error(f"Model {model} is not supported in TorchTune.")

        # Save the state dict
        if output_path is None:
            checkpoint_dir = checkpoint_path.parent
            output_path = checkpoint_dir / PYTORCH_MODEL_FILENAME

        output_state_dict = {}
        if train_type == "lora":
            output_state_dict[MODEL_KEY] = state_dict
        else:
            output_state_dict = state_dict
        torch.save(output_state_dict, output_path)

        self._logger.info(
            msg=f"Succesfully wrote PyTorch-native model checkpoint to {output_path}"
        )

    def parse_args(self) -> argparse.Namespace:
        args = self._parser.parse_args()
        if args.func == _torchrun_cmd:
            # Point UUID to the actual recipe and config file
            recipe_spec = args.recipe
            if recipe_spec in list_recipes():
                args.recipe = str(ROOT / "recipes" / f"{recipe_spec}.py")
                config_idx = args.recipes_args.index("--config") + 1
                config_spec = args.recipes_args[config_idx]
                if config_spec in list_configs(recipe_spec):
                    args.recipes_args[config_idx] = str(
                        ROOT / "recipes" / "configs" / f"{config_uuid}.yaml"
                    )

            # If the user is running `tune run`, we need to reset the `recipe` and `recipe_args`
            args.__dict__["training_script"] = args.__dict__.pop("recipe")
            args.__dict__["training_script_args"] = args.__dict__.pop("recipe_args")
        return args

    def run(self, args: argparse.Namespace):
        """Execute CLI"""
        args.func(args)


def main():
    """Entrypoint for CLI"""
    parser = TuneArgumentParser()
    args = parser.parse_args()
    parser.run(args)
