#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys

from tests.common import TUNE_PATH


class TestTuneCLI:
    def test_tune_without_args_returns_help(self, capsys, monkeypatch):
        testargs = ["tune"]

        monkeypatch.setattr(sys, "argv", testargs)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")

        assert "Welcome to the torchtune CLI!" in output
