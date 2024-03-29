# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import List

from torchtune import datasets, models, modules, utils


@dataclass
class Config:
    name: str
    file_path: str


@dataclass
class Recipe:
    name: str
    file_path: str
    configs: List[Config]
    supports_distributed: bool

    def get_configs(self) -> List[Config]:
        return self.configs


_ALL_RECIPES = [
    Recipe(
        name="full_finetune_single_device",
        file_path="full_finetune_single_device.py",
        configs=[
            Config(
                name="llama2/7B_full_single_device",
                file_path="llama2/7B_full_single_device.yaml",
            ),
            Config(
                name="llama2/7B_full_single_device_low_memory",
                file_path="llama2/7B_full_single_device_low_memory.yaml",
            ),
        ],
        supports_distributed=False,
    ),
    Recipe(
        name="full_finetune_distributed",
        file_path="full_finetune_distributed.py",
        configs=[
            Config(name="llama2/7B_full", file_path="llama2/7B_full.yaml"),
            Config(name="llama2/13B_full", file_path="llama2/13B_full.yaml"),
        ],
        supports_distributed=True,
    ),
    Recipe(
        name="lora_finetune_single_device",
        file_path="lora_finetune_single_device.py",
        configs=[
            Config(
                name="llama2/7B_lora_single_device",
                file_path="llama2/7B_lora_single_device.yaml",
            ),
            Config(
                name="llama2/7B_qlora_single_device",
                file_path="llama2/7B_qlora_single_device.yaml",
            ),
        ],
        supports_distributed=False,
    ),
    Recipe(
        name="lora_finetune_distributed",
        file_path="lora_finetune_distributed.py",
        configs=[
            Config(name="llama2/7B_lora", file_path="llama2/7B_lora.yaml"),
            Config(name="llama2/13B_lora", file_path="llama2/13B_lora.yaml"),
        ],
        supports_distributed=True,
    ),
    Recipe(
        name="alpaca_generate",
        file_path="alpaca_generate.py",
        configs=[
            Config(name="alpaca_generate", file_path="alpaca_generate.yaml"),
        ],
        supports_distributed=False,
    ),
    Recipe(
        name="eleuther_eval",
        file_path="eleuther_eval.py",
        configs=[
            Config(name="eleuther_eval", file_path="eleuther_eval.yaml"),
        ],
        supports_distributed=False,
    ),
]


def get_all_recipes():
    """List of recipes available from the CLI."""
    return _ALL_RECIPES


__all__ = [datasets, models, modules, utils]
