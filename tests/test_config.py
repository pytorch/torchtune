# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import importlib
import inspect

from pathlib import Path
from unittest.mock import patch

import pytest

import torch
import torchtune

from omegaconf import OmegaConf

from tests.test_utils import gpu_test, TOKENIZER_PATHS
from torchtune import config, training
from torchtune._recipe_registry import get_all_recipes
from torchtune.config._utils import _merge_yaml_and_cli_args

ROOT = Path(torchtune.__file__).parent.parent

RECIPE_ROOT = ROOT / "recipes"
CONFIG_ROOT = ROOT / "recipes" / "configs"


def _get_all_configs(recipe_types):
    """
    Flattens all recipes and configs.
    """
    recipes = get_all_recipes()
    return [
        (recipe.file_path, config.file_path)
        for recipe in recipes
        if recipe.name in recipe_types
        for config in recipe.configs
    ]


def load_module_from_path(module_name, path):
    """
    The recipes/ module is not importable, so this is a
    bit of a hack so we can actually get the relevant recipe
    class from the file. This means we can directly instantiate
    the recipe rather than dispatch through runpy.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRecipeConfigs:
    def validate_tokenizer(self, tokenizer_cfg):
        with pytest.raises(OSError, match="No such file or directory"):
            config.instantiate(tokenizer_cfg)

    def validate_checkpointer(self, checkpointer_cfg):
        checkpointer_class = torchtune.config._utils._get_component_from_path(
            checkpointer_cfg["_component_"]
        )
        with patch(
            "torchtune.training.checkpointing._checkpointer.get_path",
            return_value="boo",
        ):
            if checkpointer_class.__name__ == "FullModelHFCheckpointer":
                with pytest.raises(OSError, match="No such file or directory"):
                    config.instantiate(checkpointer_cfg)
            else:
                config.instantiate(checkpointer_cfg)

    @pytest.mark.integration_test
    @gpu_test(gpu_count=1)
    @pytest.mark.parametrize(
        "recipe_file_path, config_file_path",
        _get_all_configs(
            ["full_finetune_single_device", "lora_finetune_single_device"]
        ),
    )
    @patch("torchtune.datasets._sft.load_dataset")
    def test_single_device_recipe_config_setup(
        self, load_dataset, recipe_file_path, config_file_path, tmpdir, monkeypatch
    ):
        recipe_file_path = RECIPE_ROOT / recipe_file_path
        config_file_path = CONFIG_ROOT / config_file_path

        # We need to find the recipe object from the .py file, e.g. FullFinetuneRecipeSingleDevice
        # from full_finetune_single_device.py

        # First, let's directly load the recipe .py
        module = load_module_from_path("recipe_module", recipe_file_path)

        recipe_class = None
        # inspect is used here to find all of the available members in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if "Recipe" in name and "Interface" not in name:
                # We now have a reference to the recipe class itself!
                recipe_class = obj
                break

        assert recipe_class is not None

        # We're pulling out a lot of the config parsing logic we use to run recipes
        # from the CLI here
        parser = config._parse.TuneRecipeArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        yaml_args, cli_args = parser.parse_known_args(
            args=["--config"] + [str(config_file_path)]
        )
        cfg = _merge_yaml_and_cli_args(yaml_args, cli_args)

        assert cfg.device == "cuda"
        if cfg.get("enable_activation_offloading", False):
            cfg.pop("enable_activation_offloading")

        cfg.device = "meta"
        cfg.output_dir = str(tmpdir)

        # We need to validate disk-state dependent components separately and mock
        # them in the recipe itself
        self.validate_checkpointer(cfg.checkpointer)
        self.validate_tokenizer(cfg.tokenizer)

        # Slight hack since `optimizer.fused` will complain with meta device
        if "fused" in cfg.optimizer:
            cfg.optimizer.pop("fused")

        # Now let's mock the tokenizer with an arbitrary tokenizer we use for other
        # integration tests and is available when we download test artefacts
        cfg.tokenizer = OmegaConf.create(
            {
                "_component_": "torchtune.models.llama3.llama3_tokenizer",
                "path": TOKENIZER_PATHS["llama3"],
            }
        )

        # Let's grab the model state dict so we can mock the `load_checkpoint`
        # function in the recipe. We do this by instantiating the configured model
        # on meta device, pulling out the state dict, and mocking the recipe function return
        # value to use this state dict.
        with torch.device("meta"):
            # If we're using a model with LoRA/QLoRA enabled
            # we need to prune any LoRA weights from the state dict
            # (because we're always loading base model weights).
            # We also need to mock `_register_state_dict_hook` for QLoRA models
            # as this hook will attempt to copy model weights, which don't actually
            # exist on meta device.
            if "lora" in cfg.model._component_:
                with patch.object(
                    torchtune.modules.TransformerDecoder,
                    "_register_state_dict_hook",
                    return_value=None,
                ):
                    model = config.instantiate(cfg.model)
                    state_dict = {
                        k: v for k, v in model.state_dict().items() if "lora" not in k
                    }
            else:
                state_dict = config.instantiate(cfg.model).state_dict()

        # mock any calls to `datasets.load_dataset` since we don't actually want to
        # load any datasets in - this is expensive!
        load_dataset.return_value = [0]
        with patch.object(
            recipe_class,
            "load_checkpoint",
            return_value={training.MODEL_KEY: state_dict},
        ):
            recipe = recipe_class(cfg=cfg)
            recipe.setup(cfg)

        assert True
