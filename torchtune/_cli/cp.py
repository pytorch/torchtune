# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This script copies a built-in recipe or config to a local path."""
import shutil
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


def cp_cmd(parser, *other_args):
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
