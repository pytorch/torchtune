#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys
from unittest.mock import patch

import pytest
from recipes import list_configs, list_recipes

tune_path = "scripts/cli_utils/tune"


class TestTuneCLI:
    def test_recipe_list(self, capsys):
        testargs = "tune recipe list".split()
        with patch.object(sys, "argv", testargs):
            runpy.run_path(tune_path, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n").split("\n")
        assert (
            output == list_recipes()
        ), "Output must match recipe list from recipes/__init__.py"

    def test_recipe_cp(self, tmp_path, capsys):
        # Valid recipe
        recipe = "finetune_llm"
        path = tmp_path / "dummy.py"
        testargs = f"tune recipe cp -r {recipe} -p {path}".split()
        with patch.object(sys, "argv", testargs):
            runpy.run_path(tune_path, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")
        assert output == f"Copied recipe {recipe} to {path}"

        # File exists error
        with patch.object(sys, "argv", testargs):
            runpy.run_path(tune_path, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")
        assert output == f"File already exists at {path}, not overwriting"

        # Invalid recipe error
        recipe = "fake"
        path = tmp_path / "dummy2.py"
        testargs = f"tune recipe cp -r {recipe} -p {path}".split()
        with patch.object(sys, "argv", testargs):
            runpy.run_path(tune_path, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")
        assert output == f"Invalid recipe name {recipe} provided, no such recipe"

    def test_config_list(self, capsys):
        recipe = "finetune_llm"
        testargs = f"tune config list --recipe {recipe}".split()
        with patch.object(sys, "argv", testargs):
            runpy.run_path(tune_path, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n").split("\n")
        assert output == list_configs(
            recipe
        ), "Output must match config list from recipes/__init__.py"

    def test_config_cp(self, tmp_path, capsys):
        # Valid recipe
        config = "alpaca_llama2_finetune"
        path = tmp_path / "dummy.yaml"
        testargs = f"tune config cp -c {config} -p {path}".split()
        with patch.object(sys, "argv", testargs):
            runpy.run_path(tune_path, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")
        assert output == f"Copied config {config} to {path}"

        # File exists error
        with patch.object(sys, "argv", testargs):
            runpy.run_path(tune_path, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")
        assert output == f"File already exists at {path}, not overwriting"

        # Invalid recipe error
        config = "fake"
        path = tmp_path / "dummy2.yaml"
        testargs = f"tune config cp -c {config} -p {path}".split()
        with patch.object(sys, "argv", testargs):
            runpy.run_path(tune_path, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")
        assert output == f"Invalid config name {config} provided, no such config"

    def test_run(self, capsys):
        recipe = "finetune_llm"
        testargs = (
            f"tune {recipe} --config alpaca_llama2_finetune --tokenizer fake".split()
        )
        with patch.object(sys, "argv", testargs):
            with pytest.raises(SystemExit) as e:
                runpy.run_path(tune_path, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.err.rstrip("\n").split("\n")[-1]
        assert (
            output
            == "finetune_llm.py: error: argument --tokenizer: invalid choice: 'fake' (choose from 'llama2_tokenizer')"
        )
