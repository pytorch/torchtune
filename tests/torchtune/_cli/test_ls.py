# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import runpy
import sys

from tests.common import TUNE_PATH

from torchtune._recipe_registry import get_all_recipes


class TestTuneListCommand:
    """This class tests the `tune ls` command."""

    def test_ls_lists_all_recipes_and_configs(self, capsys, monkeypatch):
        testargs = "tune ls".split()

        monkeypatch.setattr(sys, "argv", testargs)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")

        for recipe in get_all_recipes(include_experimental=False):
            assert recipe.name in output
            for config in recipe.configs:
                assert config.name in output

    def test_ls_lists_all_recipes_and_configs_experimental(self, capsys, monkeypatch):
        testargs = "tune ls --experimental".split()

        monkeypatch.setattr(sys, "argv", testargs)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")

        for recipe in get_all_recipes():
            assert recipe.name in output
            for config in recipe.configs:
                assert config.name in output
