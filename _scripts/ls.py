# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This script lists all built-in recipes/configs"""

import argparse
import textwrap

from recipes import list_configs, list_recipes


def main():
    # Print table header
    header = "{:<20} {:<15}".format("RECIPE", "CONFIG")
    print(header)

    # Print recipe/config pairs
    for recipe in list_recipes():
        configs = list_configs(recipe)
        # If there are no configs for a recipe, print a blank config
        if len(configs) == 0:
            row = "{:<20} {:<15}".format(recipe, "<>")
            print(row)
        for i, config in enumerate(configs):
            # If there are multiple configs for a single recipe, omit the recipe name
            # on latter configs
            if i > 0:
                recipe = ""
            row = "{:<20} {:<15}".format(recipe, config)
            print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List all built-in recipes and configs",
        usage="tune ls",
        epilog=textwrap.dedent(
            """\
        examples:
            $ tune ls
            RECIPE               CONFIG
            full_finetune        alpaca_llama2_full_finetune
            lora_finetune        alpaca_llama2_lora_finetune
            alpaca_generate      <>

        To run one of these recipes:
            $ tune full_finetune --config alpaca_llama2_full_finetune
        """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.parse_args()
    main()
