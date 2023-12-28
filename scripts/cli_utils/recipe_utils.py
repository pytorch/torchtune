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

from recipes import list_recipes


def print_recipes():
    print(*list_recipes(), sep="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility for information relating to recipes"
    )
    subparsers = parser.add_subparsers(metavar="")

    # Parser for list command
    list_parser = subparsers.add_parser("list", help="List recipes")
    list_parser.set_defaults(func=print_recipes)

    # Parse arguments and run command
    args = parser.parse_args()
    args.func()
