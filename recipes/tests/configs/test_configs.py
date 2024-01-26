# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os

from omegaconf import OmegaConf
from recipes.finetune_llm import recipe as finetune_llm_recipe
from torchtune.utils.config import validate_recipe_args

ROOT_DIR: str = os.path.join(os.path.abspath(__file__), "../../../configs")

config_to_recipe = {
    os.path.join(ROOT_DIR, "alpaca_llama2_finetune.yaml"): finetune_llm_recipe,
}


class TestConfigs:
    """Tests that all configs are well formed.
    Configs should have the complete set of arguments as specified by the recipe.
    """

    def test_configs(self) -> None:
        for config_path, recipe in config_to_recipe.items():
            args = OmegaConf.load(config_path)
            try:
                validate_recipe_args(recipe, args)
            except (TypeError, ValueError) as e:
                raise AssertionError(
                    f"Config {config_path} for recipe {recipe.__name__} is not well formed"
                ) from e
