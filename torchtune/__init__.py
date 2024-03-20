# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune import datasets, models, modules, utils

_RECIPE_LIST = [
    "full_finetune_single_device.py",
    "full_finetune_distributed.py",
    "alpaca_generate.py",
    "lora_finetune_single_device.py",
    "lora_finetune_distributed.py",
]
_CONFIG_LISTS = {
    "full_finetune_single_device.py": ["full_finetune_single_device.yaml"],
    "full_finetune_distributed.py": ["full_finetune_distributed.yaml"],
    "lora_finetune_single_device.py": ["lora_finetune_single_device.yaml"],
    "lora_finetune_distributed.py": ["lora_finetune_distributed.yaml"],
    "alpaca_generate.py": ["alpaca_generate.yaml"],
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
