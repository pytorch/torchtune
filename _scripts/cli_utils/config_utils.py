# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Utility for getting and displaying information on packaged configs.

"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import torchtune
from recipes import list_configs


def print_help(parser):
    def func(args):
        parser.print_help(sys.stderr)
        sys.exit(1)

    return func


def print_configs(args):
    recipe_name = args.recipe
    print(*list_configs(recipe_name), sep="\n")


def get_config_path(name):
    pkg_path = str(Path(torchtune.__file__).parent.parent.absolute())
    config_path = os.path.join(pkg_path, "recipes", "configs", f"{name}.yaml")
    return config_path


def copy_config(args):
    config = args.config
    path = args.path
    if path is None:
        path = os.path.join(os.getcwd(), f"{config}.yaml")

    # Get config path
    config_path = get_config_path(config)

    # Copy config
    try:
        if not os.path.exists(path):
            shutil.copyfile(config_path, path)
        else:
            raise FileExistsError
        print(f"Copied config {config} to {path}")
    except FileNotFoundError:
        print(f"Invalid config name {config} provided, no such config")
    except FileExistsError:
        print(f"File already exists at {path}, not overwriting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility for information relating to recipe configs",
        usage="tune config",
    )
    parser.set_defaults(func=print_help(parser))
    subparsers = parser.add_subparsers(metavar="")

    # Parser for list command
    list_parser = subparsers.add_parser("list", help="List configs")
    list_parser.add_argument(
        "--recipe",
        "-r",
        type=str,
        required=True,
        help="Name of recipe to retrieve configs for",
    )
    list_parser.set_defaults(func=print_configs)

    # Parser for copy command
    list_parser = subparsers.add_parser("cp", help="Copy config to local path")
    list_parser.add_argument("config", type=str, help="Name of config to copy")
    list_parser.add_argument(
        "path",
        nargs="?",
        default=None,
        type=str,
        help="Path to copy config to (defaults to current directory and name)",
    )
    list_parser.set_defaults(func=copy_config)

    # Parse arguments and run command
    args = parser.parse_args()
    args.func(args)
