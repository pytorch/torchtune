#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import torchtune

from torchtune import list_configs, list_recipes


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
