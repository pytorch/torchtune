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
from torchtune import list_configs, list_recipes


def _get_absolute_path(file_name: str) -> Path:
    pkg_path = Path(torchtune.__file__).parent.parent.absolute()
    recipes_path = pkg_path / "recipes"
    if file_name.endswith(".yaml"):
        path = recipes_path / "configs" / file_name
    else:
        assert file_name.endswith(".py"), f"Expected .py file, got {file_name}"
        path = recipes_path / file_name
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
            f"Invalid file name: {args.file}. Try `tune ls` to see all available files to copy."
        )

    # Get file path
    file_name = args.file
    src = _get_absolute_path(file_name)

    # Copy file
    try:
        if args.no_clobber and destination.exists():
            print(f"File already exists at {destination.absolute()}, not overwriting.")
        else:
            if args.make_parents:
                destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, destination)
    except FileNotFoundError:
        parser.error(
            f"Cannot create regular file: '{destination}'. No such file or directory. "
            "If the specified destination's parent directory does not exist and you would "
            "like to create it on-the-fly, use the --make-parents flag."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="tune cp",
        usage="tune cp <recipe|config> destination [OPTIONS]",
        description="Copy a built-in recipe or config to a local path.",
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
    parser.add_argument(
        "file",
        type=str,
        help="Recipe/config to copy. For a list of all possible options, run `tune ls`",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Location to copy the file to",
    )
    parser.add_argument(
        "-n",
        "--no-clobber",
        action="store_true",
        help="Do not overwrite destination if it already exists",
        default=False,
    )
    parser.add_argument(
        "--make-parents",
        action="store_true",
        help="Create parent directories for destination if they do not exist. "
        "If not set to True, will error if parent directories do not exist",
        default=False,
    )
    main(parser)
