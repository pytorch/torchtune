# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune import datasets, models, modules, utils

_RECIPE_LIST = ["full_finetune.py", "lora_finetune.py", "alpaca_generate.py"]
_CONFIG_LISTS = {
    "full_finetune.py": ["alpaca_llama2_full_finetune.yaml"],
    "lora_finetune.py": ["alpaca_llama2_lora_finetune.yaml"],
    "alpaca_generate.py": ["alpaca_llama2_generate.yaml"],
}


def list_recipes():
    """List of recipes available from the CLI"""
    return _RECIPE_LIST


def list_configs(recipe: str):
    """List of configs available from the CLI given a recipe"""
    if recipe not in _CONFIG_LISTS:
        raise ValueError(f"Unknown recipe: {recipe}")
    return _CONFIG_LISTS[recipe]


__all__ = [datasets, models, modules, utils]
