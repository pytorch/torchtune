# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from pathlib import Path
from typing import List

import torchtune

_CONFIG_LISTS = {
    "full_finetune": ["alpaca_llama2_full_finetune"],
    "lora_finetune": ["alpaca_llama2_lora_finetune"],
    "alpaca_generate": [],
}


def list_recipes() -> List[str]:
    """List of available recipes available from the CLI"""
    not_recipes = {
        "__init__.py",
        "interfaces.py",
        "params.py",
    }
    pkg_path = str(Path(torchtune.__file__).parent.parent.absolute())
    python_recipes = [
        f for f in os.listdir(os.path.join(pkg_path, "recipes")) if f.endswith(".py")
    ]
    return [f[:-3] for f in python_recipes if f not in not_recipes]


def list_configs(recipe: str) -> List[str]:
    """List of availabe configs available from the CLI given a recipe"""
    if recipe not in _CONFIG_LISTS:
        raise ValueError(f"Unknown recipe: {recipe}")
    return _CONFIG_LISTS[recipe]
