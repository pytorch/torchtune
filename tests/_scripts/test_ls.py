#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys
from unittest.mock import patch

from recipes import list_configs, list_recipes

from tests._scripts.common import TUNE_PATH


class TestTuneCLIWithListScript:
    def test_ls_lists_all_models(self, capsys):
        testargs = "tune ls".split()
        with patch.object(sys, "argv", testargs):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")
        for recipe in list_recipes():
            assert recipe in output, f"{recipe} was not found in output"
            for config in list_configs(recipe):
                assert config in output, f"{config} was not found in output"
