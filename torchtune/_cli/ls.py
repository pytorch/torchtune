# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This script lists all built-in recipes/configs"""

from torchtune import list_configs, list_recipes

_NULL_VALUE = "<>"


def ls_cmd(args):
    # Print table header
    header = f"{'RECIPE':<40} {'CONFIG':<40}"
    print(header)

    # Print recipe/config pairs
    for recipe in list_recipes():
        configs = list_configs(recipe)
        # If there are no configs for a recipe, print a blank config
        if len(configs) == 0:
            row = f"{recipe:<40} {_NULL_VALUE:<40}"
            print(row)
        for i, config in enumerate(configs):
            # If there are multiple configs for a single recipe, omit the recipe name
            # on latter configs
            if i > 0:
                recipe = ""
            row = f"{recipe:<40} {config:<40}"
            print(row)
