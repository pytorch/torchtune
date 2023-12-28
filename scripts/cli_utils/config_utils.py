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

import argparse

from recipes import list_configs


def print_configs(args):
    recipe_name = args.recipe
    print(*list_configs(recipe_name), sep="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility for information relating to recipe configs"
    )
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

    # Parse arguments and run command
    args = parser.parse_args()
    args.func(args)
