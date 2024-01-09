# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Utility for getting and displaying information on packaged recipes.

"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import torchtune
from recipes import list_recipes


def print_help(parser):
    def func(args):
        parser.print_help(sys.stderr)
        sys.exit(1)

    return func


def print_recipes(args):
    print(*list_recipes(), sep="\n")


def get_recipe_path(name):
    pkg_path = str(Path(torchtune.__file__).parent.parent.absolute())
    recipe_path = os.path.join(pkg_path, "recipes", f"{name}.py")
    return recipe_path


def copy_recipe(args):
    recipe = args.recipe
    path = args.path
    if path is None:
        path = os.path.join(os.getcwd(), f"{recipe}.py")

    # Get recipe path
    recipe_path = get_recipe_path(recipe)

    # Copy recipe
    try:
        if not os.path.exists(path):
            shutil.copyfile(recipe_path, path)
        else:
            raise FileExistsError
        print(f"Copied recipe {recipe} to {path}")
    except FileNotFoundError:
        print(f"Invalid recipe name {recipe} provided, no such recipe")
    except FileExistsError:
        print(f"File already exists at {path}, not overwriting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility for information relating to recipes", usage="tune recipe"
    )
    parser.set_defaults(func=print_help(parser))
    subparsers = parser.add_subparsers(metavar="")

    # Parser for list command
    list_parser = subparsers.add_parser("list", help="List recipes")
    list_parser.set_defaults(func=print_recipes)

    # Parser for copy command
    list_parser = subparsers.add_parser("cp", help="Copy recipe to local path")
    list_parser.add_argument("recipe", type=str, help="Name of recipe to copy")
    list_parser.add_argument(
        "path",
        nargs="?",
        default=None,
        type=str,
        help="Path to copy recipe to (defaults to current directory and name)",
    )
    list_parser.set_defaults(func=copy_recipe)

    # Parse arguments and run command
    args = parser.parse_args()
    args.func(args)
