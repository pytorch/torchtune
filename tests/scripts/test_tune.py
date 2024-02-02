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

from tests.scripts.common import TUNE_PATH


class TestTuneCLI:
    def test_recipe_list(self, capsys):
        testargs = "tune recipe list".split()
        with patch.object(sys, "argv", testargs):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n").split("\n")
        assert (
            output == list_recipes()
        ), "Output must match recipe list from recipes/__init__.py"

    def test_recipe_cp(self, tmp_path, capsys):
        # Valid recipe
        recipe = "finetune_llm"
        path = tmp_path / "dummy.py"
        testargs = f"tune recipe cp {recipe} {path}".split()
        with patch.object(sys, "argv", testargs):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")
        assert output == f"Copied recipe {recipe} to {path}"

        # File exists error
        with patch.object(sys, "argv", testargs):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")
        assert output == f"File already exists at {path}, not overwriting"

        # Invalid recipe error
        recipe = "fake"
        path = tmp_path / "dummy2.py"
        testargs = f"tune recipe cp {recipe} {path}".split()
        with patch.object(sys, "argv", testargs):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")
        assert output == f"Invalid recipe name {recipe} provided, no such recipe"

    def test_recipe_paths(self):
        recipes = list_recipes()
        for recipe in recipes:
            pkg_path = str(Path(torchtune.__file__).parent.parent.absolute())
            recipe_path = os.path.join(pkg_path, "recipes", f"{recipe}.py")
            assert os.path.exists(recipe_path), f"{recipe_path} must exist"

    def test_config_list(self, capsys):
        recipe = "finetune_llm"
        testargs = f"tune config list --recipe {recipe}".split()
        with patch.object(sys, "argv", testargs):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n").split("\n")
        assert output == list_configs(
            recipe
        ), "Output must match config list from recipes/__init__.py"

    def test_config_cp(self, tmp_path, capsys):
        # Valid recipe
        config = "alpaca_llama2_finetune"
        path = tmp_path / "dummy.yaml"
        testargs = f"tune config cp {config} {path}".split()
        with patch.object(sys, "argv", testargs):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")
        assert output == f"Copied config {config} to {path}"

        # File exists error
        with patch.object(sys, "argv", testargs):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")
        assert output == f"File already exists at {path}, not overwriting"

        # Invalid recipe error
        config = "fake"
        path = tmp_path / "dummy2.yaml"
        testargs = f"tune config cp {config} {path}".split()
        with patch.object(sys, "argv", testargs):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")
        assert output == f"Invalid config name {config} provided, no such config"

    def test_config_paths(self):
        recipes = list_recipes()
        for recipe in recipes:
            configs = list_configs(recipe)
            for config in configs:
                pkg_path = str(Path(torchtune.__file__).parent.parent.absolute())
                config_path = os.path.join(
                    pkg_path, "recipes", "configs", f"{config}.yaml"
                )
                assert os.path.exists(config_path), f"{config_path} must exist"

    def test_run(self, capsys):
        recipe = "finetune_llm"
        # Make sure we're not running on GPU which can lead to issues on GH CI
        testargs = f"\
            tune {recipe} --config alpaca_llama2_finetune --override tokenizer=fake \
            device=cpu enable_fsdp=False enable_activation_checkpointing=False \
        ".split()
        with patch.object(sys, "argv", testargs):
            # TODO: mock recipe so we don't actually run it,
            # we purposely error out prematurely so we can just test that we
            # enter the script successfully
            with pytest.raises(ValueError):
                runpy.run_path(TUNE_PATH, run_name="__main__")
