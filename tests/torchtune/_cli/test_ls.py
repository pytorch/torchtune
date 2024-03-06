#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys

from tests.common import TUNE_PATH

from torchtune import list_configs, list_recipes

from torchtune._cli.ls import _NULL_VALUE


class TestTuneCLIWithListScript:
    def test_ls_lists_all_models(self, capsys, monkeypatch):
        testargs = "tune ls".split()

        monkeypatch.setattr(sys, "argv", testargs)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")
        for recipe in list_recipes():
            assert recipe in output, f"{recipe} was not found in output"
            all_configs = list_configs(recipe)
            if len(all_configs) > 0:
                for config in list_configs(recipe):
                    assert config in output, f"{config} was not found in output"
            else:
                assert _NULL_VALUE in output, f"{_NULL_VALUE} was not found in output"
