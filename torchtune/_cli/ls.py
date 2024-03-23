# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This script lists all built-in recipes/configs"""

import argparse
import textwrap

from torchtune import list_configs, list_recipes

_NULL_VALUE = "<>"


def main():
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List all built-in recipes and configs",
        usage="tune ls",
        epilog=textwrap.dedent(
            """\
        examples:
            $ tune ls
            RECIPE                           CONFIG
            full_finetune_distributed.py     llama2/7B_full, llama2/13B_full
            lora_finetune_distributed.py     llama2/7B_lora, llama2/13B_lora
            alpaca_generate.py               alpaca_generate.yaml

        To run one of these recipes:
            $ tune full_finetune_single_device --config llama2/7B_full_single_device
        """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.parse_args()
    main()
