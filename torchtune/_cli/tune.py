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

# """
# Launcher and utilities for torchtune recipes

# ``tune`` provides functionality for launching torchtune recipes as well as local
# recipes. Aside from torchtune recipe utilties it integrates with ``torch.distributed.run``
# to support distributed job launching by default. ``tune`` offers everyting that ``torchrun``
# does with the following additional functionalities:

# 1. ``tune <recipe> <recipe_args>`` with no optional ``torchrun`` options launches a single python process

# 2. ``<recipe>`` and recipe arg ``<config>`` can both be passed in as names instead of paths if they're included in torchtune

# 3. ``tune <path/to/recipe.py> <recipe_args>`` can be used to launch local recipes

# 4. ``tune <torchrun_options> <recipe> <recipe_args>`` will launch a torchrun job

# .. note:: ``tune`` is a python
#           `console script <https://packaging.python.org/en/latest/specifications/entry-points/#use-for-scripts>`_
#           to the main module
#           `scripts.cli_utils.tune <https://github.com/pytorch/torchtune/blob/main/scripts/cli_utils/tune>`_
#           declared in the ``scripts`` configuration in
#           `setup.py <https://github.com/pytorch/torchtune/blob/main/setup.py>`_.
#           It is equivalent to invoking ``python -m scripts.cli_utils.tune``.
# """
# import argparse
# import runpy
# import sys
# from pathlib import Path

# import torchtune
# from torchtune import list_recipes
# from torchtune._cli import list_scripts
# from torchtune.utils._distributed import _valid_distributed_single_node_nnodes


# def _update_parser_help(parser):
#     parser.description = "Torch Tune Recipe Launcher"
#     parser.usage = "tune [options] <recipe> [recipe_args]"
#     parser.formatter_class = argparse.RawDescriptionHelpFormatter

#     # Update torchrun argparse name for more accurate CLI help
#     actions = [a.dest for a in parser._actions]
#     # Update training_script help to be recipe
#     idx = actions.index("training_script")
#     parser._actions[idx].dest = "recipe"
#     parser._actions[idx].help = "Name or path to recipe to be launched followed by args"

#     # Update training_script_args help to be recipe_args
#     idx = actions.index("training_script_args")
#     parser._actions[idx].dest = "recipe_args"


# def _is_distributed_args(args):
#     total = len(sys.argv) - 1  # total args minus "tune"
#     script_args = len(args.recipe_args) + 1  # script args + 1 for script name
#     return total > script_args


#     # TODO (rohan-varma): Add check that nproc_per_node <= cuda device count. Currently,
#     # we don't do this since we test on CPUs for distributed. Will update once multi GPU
#     # CI is supported.

import argparse
import os
import textwrap
from pathlib import Path

from torch.distributed.run import (
    get_args_parser as get_torchrun_args_parser,
    run as torchrun_cmd,
)
from torchtune._cli.convert_checkpoint import (
    _PYTORCH_MODEL_FILENAME,
    convert_checkpoint_cmd,
)

from torchtune._cli.cp import cp_cmd
from torchtune._cli.download import download_cmd
from torchtune._cli.ls import ls_cmd
from torchtune._cli.validate import validate_cmd


def main():
    tune_parser = argparse.ArgumentParser(
        prog="tune",
        description="Welcome to the TorchTune CLI!",
        add_help=True,
    )
    tune_parser.set_defaults(func=lambda args: tune_parser.print_help())
    subparsers = tune_parser.add_subparsers(title="subcommands")

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
    ls_parser.set_defaults(func=ls_cmd)

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
    cp_parser.set_defaults(func=cp_cmd)

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
    download_parser.set_defaults(func=download_cmd)

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
        f"under the filename {_PYTORCH_MODEL_FILENAME}.",
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
    convert_ckpt_parser.set_defaults(func=convert_checkpoint_cmd)

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
    validate_parser.set_defaults(func=validate_cmd)

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

    torchrun_argparser = get_torchrun_args_parser()
    for action in torchrun_argparser._actions:
        if action.dest == "nproc_per_node":
            action.option_strings += ["--num-gpu", "--num_gpu"]
        elif action.dest == "training_script":
            action.dest = "recipe"
            action.help = "Name or path to recipe to be launched followed by args. For a list of all possible recipes, run `tune ls`."
        elif action.dest == "training_script_args":
            action.dest = "recipe_args"
            action.help = "Args to be passed to the recipe."
        elif action.dest == "help":
            continue
        try:
            run_parser._add_action(action)
        except Exception as e:
            print(e)
            import pdb

            pdb.set_trace()
    run_parser.set_defaults(func=torchrun_cmd)

    args = tune_parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
