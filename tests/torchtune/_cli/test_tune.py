#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torchtune
from recipes import list_configs, list_recipes

from tests.torchtune._cli.common import TUNE_PATH


class TestTuneCLI:
    def test_recipe_paths(self):
        recipes = list_recipes()
        for recipe in recipes:
            pkg_path = str(Path(torchtune.__file__).parent.parent.absolute())
            recipe_path = os.path.join(pkg_path, "recipes", recipe)
            assert os.path.exists(recipe_path), f"{recipe_path} must exist"

    def test_config_paths(self):
        recipes = list_recipes()
        for recipe in recipes:
            configs = list_configs(recipe)
            for config in configs:
                pkg_path = str(Path(torchtune.__file__).parent.parent.absolute())
                config_path = os.path.join(pkg_path, "recipes", "configs", config)
                assert os.path.exists(config_path), f"{config_path} must exist"

    def test_run(self, capsys):
        recipe = "full_finetune"
        # Make sure we're not running on GPU which can lead to issues on GH CI
        testargs = f"\
            tune {recipe} --config alpaca_llama2_full_finetune --override tokenizer=fake \
            device=cpu enable_fsdp=False enable_activation_checkpointing=False \
            model_checkpoint=/tmp/fake.pt \
        ".split()
        with patch.object(sys, "argv", testargs):
            # TODO: mock recipe so we don't actually run it,
            # we purposely error out prematurely so we can just test that we
            # enter the script successfully
            with pytest.raises(FileNotFoundError, match="No such file or directory"):
                runpy.run_path(TUNE_PATH, run_name="__main__")
