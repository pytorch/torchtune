# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import functools
import sys
from argparse import Namespace
from typing import Any, Callable, List, Tuple

from omegaconf import DictConfig, OmegaConf
from torchtune import training
from torchtune.config._utils import _merge_yaml_and_cli_args

import os
from unittest.mock import patch
import runpy

import sys
from pathlib import Path

import numpy as np
import torchtune

import pytest

import torch
from tests.common import TUNE_PATH

from tests.recipes.utils import (
    CKPT_COMPONENT_MAP,
    dummy_alpaca_dataset_config,
    MODEL_TEST_CONFIGS,
    write_hf_ckpt_config,
)
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    gen_log_file_name,
    get_loss_values_from_metric_logger,
    TOKENIZER_PATHS,
    gpu_test
)
from torchtune._recipe_registry import Config, get_all_recipes, Recipe
from torchtune import config
import inspect

ROOT = Path(torchtune.__file__).parent.parent

RECIPE_ROOT = ROOT / "recipes"
CONFIG_ROOT = ROOT / "recipes" / "configs"
import pytest


def _get_all_configs(recipe_types):
    recipes = get_all_recipes()
    # valid_recipes = ["full_finetune_single_device"]#, "lora_finetune_single_device"]
    return [
        (recipe.file_path, config.file_path)
        for recipe in recipes
        if recipe.name in recipe_types
        for config in recipe.configs
    ]


import importlib


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRecipeConfigs:

    def validate_tokenizer(self, tokenizer_cfg):
        with pytest.raises(OSError, match="No such file or directory"):
            config.instantiate(tokenizer_cfg)

    def validate_checkpointer(self, checkpointer_cfg):
        checkpointer_class = torchtune.config._utils._get_component_from_path(checkpointer_cfg["_component_"])
        with patch("torchtune.training.checkpointing._checkpointer.get_path", return_value="boo"):
            if checkpointer_class.__name__ == "FullModelHFCheckpointer":
                with pytest.raises(OSError, match="No such file or directory"):
                    config.instantiate(checkpointer_cfg)
            else:
                config.instantiate(checkpointer_cfg)

    @pytest.mark.integration_test
    @gpu_test(gpu_count=1)
    @pytest.mark.parametrize(
        "recipe_file_path, config_file_path",
        _get_all_configs(["full_finetune_single_device", "lora_finetune_single_device"]),
    )
    @patch("torchtune.datasets._sft.load_dataset")
    def test_single_device_full_finetune_recipe_config_setup(
        self, load_dataset, recipe_file_path, config_file_path, tmpdir, monkeypatch
    ):
        recipe_file_path = RECIPE_ROOT / recipe_file_path
        config_file_path = CONFIG_ROOT / config_file_path
        module = load_module_from_path("recipe_module", recipe_file_path)
        recipe_class = None
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if "Recipe" in name and "Interface" not in name:  # Adjust this condition based on your naming convention
                recipe_class = obj
                break

        assert recipe_class is not None

        parser = config._parse.TuneRecipeArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        # Get user-specified args from config and CLI and create params for recipe
        yaml_args, cli_args = parser.parse_known_args(args=["--config"] + [str(config_file_path)] + ["device=meta"])
        cfg = _merge_yaml_and_cli_args(yaml_args, cli_args)
        cfg.output_dir = str(tmpdir)
        self.validate_checkpointer(cfg.checkpointer)
        self.validate_tokenizer(cfg.tokenizer)
        cfg.tokenizer = OmegaConf.create(
            {"_component_": "torchtune.models.llama3.llama3_tokenizer", "path": TOKENIZER_PATHS["llama3"]}
        )

        with torch.device("meta"):
            if "lora" in cfg.model._component_:
                model_module = ".".join(cfg.model._component_.split(".")[:3])
                try:
                    with patch(model_module + "._component_builders._register_reparametrize_state_dict_hooks", return_value=None):
                        model = config.instantiate(cfg.model)
                        state_dict = {k: v for k, v in model.state_dict().items() if "lora" not in k}
                except AttributeError:
                    pass
                with patch.object(torchtune.modules.TransformerDecoder, "_register_state_dict_hook", return_value=None):
                    model = config.instantiate(cfg.model)
                    state_dict = {k: v for k, v in model.state_dict().items() if "lora" not in k}
            else:
                state_dict = config.instantiate(cfg.model).state_dict()

        load_dataset.return_value = [0]
        with patch.object(recipe_class, "load_checkpoint", return_value={training.MODEL_KEY: state_dict}):
            recipe = recipe_class(cfg=cfg)
            recipe.setup(cfg)

        assert True


# class TestRecipes:


# class TestFullFinetuneSingleDeviceRecipe:
#     def _get_test_config_overrides(self):
#         return [
#             "batch_size=8",
#             "device=cpu",
#             "dtype=fp32",
#             "enable_activation_checkpointing=False",
#             "dataset.train_on_input=False",
#             "seed=9",
#             "epochs=2",
#             "max_steps_per_epoch=2",
#             "optimizer=torch.optim.AdamW",
#             "optimizer.lr=2e-5",
#             "log_every_n_steps=1",
#             "clip_grad_norm=100",
#         ] + dummy_alpaca_dataset_config()

#     @pytest.mark.parametrize("compile", [True, False])
#     @pytest.mark.parametrize(
#         "config, model_type, ckpt_type",
#         [
#             ("llama2/7B_full_low_memory", "llama2", "meta"),
#             ("llama3/8B_full_single_device", "llama3", "tune"),
#         ],
#     )
#     def test_loss(self, compile, config, model_type, ckpt_type, tmpdir, monkeypatch):
#         ckpt_component = CKPT_COMPONENT_MAP[ckpt_type]
#         ckpt = model_type + "_" + ckpt_type
#         ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
#         tokenizer_path = Path(TOKENIZER_PATHS[model_type])
#         ckpt_dir = ckpt_path.parent
#         log_file = gen_log_file_name(tmpdir)

#         cmd = f"""
#         tune run full_finetune_single_device \
#             --config {config} \
#             output_dir={tmpdir} \
#             checkpointer._component_={ckpt_component} \
#             checkpointer.checkpoint_dir='{ckpt_dir}' \
#             checkpointer.checkpoint_files=[{ckpt_path}]\
#             checkpointer.output_dir={tmpdir} \
#             checkpointer.model_type={model_type.upper()} \
#             tokenizer.path='{tokenizer_path}' \
#             tokenizer.prompt_template=null \
#             metric_logger.filename={log_file} \
#             metric_logger.filename={log_file} \
#             metric_logger.filename={log_file} \
#             compile={compile} \
#         """.split()

#         model_config = MODEL_TEST_CONFIGS[model_type]
#         cmd = cmd + self._get_test_config_overrides() + model_config

#         monkeypatch.setattr(sys, "argv", cmd)
#         # import runpy
#         # with patch("sys.exit", return_value=None):
#         #     # Mock train and cleanup to make them no-ops
#         #     with patch("__main__.FullFinetuneRecipeSingleDevice.train", return_value=None), \
#         #          patch("__main__.FullFinetuneRecipeSingleDevice.cleanup", return_value=None):
#         #         runpy.run_path(TUNE_PATH, run_name="__main__")
#         # import pdb
#         # pdb.set_trace()
#         # with pytest.raises(SystemExit, match=""):
#         #     # Run the script and capture the module
#         loaded_module = runpy.run_path(TUNE_PATH, run_name="__main__")
#         import pdb
#         pdb.set_trace()
# with patch("__main__.FullFinetuneRecipeSingleDevice.train", return_value=None), patch(
#     "__main__.FullFinetuneRecipeSingleDevice.cleanup", return_value=None
# ):
#     with pytest.raises(SystemExit, match=""):
#         runpy.run_path(TUNE_PATH, run_name="__main__")

# Make sure to clear compile state in between tests
# if compile:
#     torch._dynamo.reset()

# loss_values = get_loss_values_from_metric_logger(log_file)
# expected_loss_values = self._fetch_expected_loss_values(model_type)

# torch.testing.assert_close(loss_values, expected_loss_values, rtol=1e-4, atol=1e-4)

# %%
