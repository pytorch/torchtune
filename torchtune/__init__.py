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
    uuid: str
    file_name: str

    def __repr___(self):
        return self.uuid


@dataclass
class Recipe:
    uuid: str
    file_name: str
    configs: List[Config]
    supports_distributed: bool

    def __repr___(self):
        return self.uuid

    def get_configs(self):
        return self.configs


_ALL_RECIPES = [
    Recipe(
        uuid="full_finetune_single_device",
        file_name="full_finetune_single_device.py",
        configs=[
            Config(
                uuid="llama2/7B_full_single_device",
                file_name="llama2/7B_full_single_device.yaml",
            ),
        ],
        supports_distributed=False,
    ),
    Recipe(
        uuid="full_finetune_distributed",
        file_name="full_finetune_distributed.py",
        configs=[
            Config(uuid="llama2/7B_full", file_name="llama2/7B_full.yaml"),
            Config(uuid="llama2/13B_full", file_name="llama2/13B_full.yaml"),
        ],
        supports_distributed=True,
    ),
    Recipe(
        uuid="lora_finetune_single_device",
        file_name="lora_finetune_single_device.py",
        configs=[
            Config(
                uuid="llama2/7B_lora_single_device",
                file_name="llama2/7B_lora_single_device.yaml",
            ),
            Config(
                uuid="llama2/7B_qlora_single_device",
                file_name="llama2/7B_qlora_single_device.yaml",
            ),
        ],
        supports_distributed=False,
    ),
    Recipe(
        uuid="lora_finetune_distributed",
        file_name="lora_finetune_distributed.py",
        configs=[
            Config(uuid="llama2/7B_lora", file_name="llama2/7B_lora.yaml"),
            Config(uuid="llama2/13B_lora", file_name="llama2/13B_lora.yaml"),
        ],
        supports_distributed=True,
    ),
    Recipe(
        uuid="alpaca_generate",
        file_name="alpaca_generate.py",
        configs=[
            Config(uuid="alpaca_generate", file_name="alpaca_generate.yaml"),
        ],
        supports_distributed=False,
    ),
    Recipe(
        uuid="eleuther_eval",
        file_name="eleuther_eval.py",
        configs=[
            Config(uuid="eleuther_eval", file_name="eleuther_eval.yaml"),
        ],
        supports_distributed=False,
    ),
]


def get_all_recipes():
    """List of recipes available from the CLI."""
    return _ALL_RECIPES


__all__ = [datasets, models, modules, utils]
