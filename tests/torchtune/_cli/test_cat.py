# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys

import pytest
from tests.common import TUNE_PATH


class TestTuneCatCommand:
    """This class tests the `tune cat` command."""

    def test_cat_valid_config(self, capsys, monkeypatch):
        testargs = "tune cat llama2/7B_full".split()
        monkeypatch.setattr(sys, "argv", testargs)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")

        # Check for key sections that should be in the YAML output
        assert "output_dir:" in output
        assert "tokenizer:" in output
        assert "model:" in output

    def test_cat_recipe_name_shows_error(self, capsys, monkeypatch):
        testargs = "tune cat full_finetune_single_device".split()
        monkeypatch.setattr(sys, "argv", testargs)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")

        assert "is a recipe, not a config" in output

    def test_cat_non_existent_config(self, capsys, monkeypatch):
        testargs = "tune cat non_existent_config".split()
        monkeypatch.setattr(sys, "argv", testargs)

        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        err = captured.err.rstrip("\n")

        assert (
            "Invalid config format: 'non_existent_config'. Must be YAML (.yaml/.yml)"
            in err
        )

    def test_cat_invalid_yaml_file(self, capsys, monkeypatch, tmpdir):
        invalid_yaml = tmpdir / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: file", encoding="utf-8")

        testargs = f"tune cat {invalid_yaml}".split()
        monkeypatch.setattr(sys, "argv", testargs)

        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        err = captured.err.rstrip("\n")

        assert "Error parsing YAML file" in err

    def test_cat_external_yaml_file(self, capsys, monkeypatch, tmpdir):
        valid_yaml = tmpdir / "external.yaml"
        valid_yaml.write_text("key: value", encoding="utf-8")

        testargs = f"tune cat {valid_yaml}".split()
        monkeypatch.setattr(sys, "argv", testargs)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")

        assert "key: value" in output
