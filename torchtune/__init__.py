# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune import datasets, models, modules, utils

_RECIPE_LIST = ["full_finetune", "lora_finetune", "alpaca_generate"]
_CONFIG_LISTS = {
    "full_finetune": ["alpaca_llama2_full_finetune"],
    "lora_finetune": ["alpaca_llama2_lora_finetune"],
    "alpaca_generate": [],
}


def list_recipes():
    """List of built-in recipes available from the CLI"""
    return _RECIPE_LIST


def list_configs(recipe: str):
    """List of built-in configs available from the CLI given a recipe"""
    if recipe not in _CONFIG_LISTS:
        raise ValueError(f"Unknown recipe: {recipe}")
    return _CONFIG_LISTS[recipe]


__all__ = [
    datasets,
    models,
    modules,
    utils,
    list_recipes,
    list_configs,
]
