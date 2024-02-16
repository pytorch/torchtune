# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This script copies a built-in recipe or config to a local path."""
import argparse
import shutil
import textwrap
from pathlib import Path

import torchtune
from recipes import list_configs, list_recipes


def _is_config_file(file_name: str) -> bool:
    return file_name.endswith(".yaml")


def _get_path(file_name: str) -> Path:
    pkg_path = Path(torchtune.__file__).parent.parent.absolute()
    recipes_path = pkg_path / "recipes"
    if _is_config_file(file_name):
        path = recipes_path / "configs" / file_name
    else:
        path = recipes_path / f"{file_name}.py"
    return path


def main(parser):
    args = parser.parse_args()
    destination = args.destination

    # Check if recipe/config is valid
    all_recipes_and_configs = list_recipes() + [
        config for recipe in list_recipes() for config in list_configs(recipe)
    ]
    if args.file not in all_recipes_and_configs:
        parser.error(
            f"Invalid file name: {args.file}. Try 'tune ls' to see all available files to copy."
        )

    # Get file path
    file_name = args.file
    src = _get_path(file_name)

    # Copy file
    try:
        if args.no_clobber and destination.exists():
            print(f"File already exists at {destination.absolute()}, not overwriting")
        else:
            shutil.copy(src, destination)
    except FileNotFoundError:
        parser.error(
            f"Cannot create regular file: '{destination}'. No such file or directory"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="tune cp",
        usage="tune cp <recipe|config> destination [OPTIONS]",
        description="Copy a built-in recipe or config to a local path.",
        epilog=textwrap.dedent(
            """\
        examples:
            $ tune cp alpaca_llama2_lora_finetune.yaml ./my_custom_llama2_lora.yaml
            $ tune cp full_finetune ./my_custom_full_finetune
        """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "file",
        type=str,
        help="Recipe or config to copy. For a list of all possible options, run `tune ls`",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Location to copy the file to",
    )
    parser.add_argument(
        "--no-clobber",
        action="store_true",
        help="Do not overwrite existing files",
        default=False,
    )
    main(parser)
