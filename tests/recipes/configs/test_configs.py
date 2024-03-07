# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os

import pytest
from omegaconf import OmegaConf
from recipes.full_finetune import FullFinetuneRecipe
from recipes.lora_finetune_distributed import LoRAFinetuneDistributedRecipe
from recipes.lora_finetune_single_device import LoRAFinetuneRecipeSingleDevice

from torchtune.utils.argparse import TuneArgumentParser

ROOT_DIR: str = os.path.join(os.path.abspath(__file__), "../../../configs")

# TODO: this probably does not scale
config_to_recipe = {
    os.path.join(ROOT_DIR, "alpaca_llama2_full_finetune.yaml"): FullFinetuneRecipe,
    os.path.join(ROOT_DIR, "alpaca_llama2_lora_finetune_distributed.yaml"): LoRAFinetuneDistributedRecipe,
    os.path.join(ROOT_DIR, "alpaca_llama2_lora_finetune_single_device.yaml"): LoRAFinetuneRecipeSingleDevice,
}


class TestConfigs:
    """Tests that all configs are well formed.
    Configs should have the complete set of arguments as specified by the recipe.
    """

    @pytest.fixture
    def parser(self):
        parser = TuneArgumentParser("Test parser")
        return parser

    # TODO: update this test to run recipes with debug args, disabling for now
    @pytest.mark.skip(
        reason="Need to update to use debug args after config system is finalized."
    )
    def test_configs(self, parser) -> None:
        for config_path, recipe in config_to_recipe.items():
            args, _ = parser.parse_known_args(["--config", config_path])
            try:
                cfg = OmegaConf.create(vars(args))
                recipe(cfg)
            except ValueError as e:
                raise AssertionError(
                    f"Config {config_path} for recipe {recipe.__name__} is not well formed"
                ) from e
