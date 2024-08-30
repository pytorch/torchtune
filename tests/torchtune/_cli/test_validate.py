# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys

import pytest
from tests.common import ASSETS, TUNE_PATH


class TestTuneValidateCommand:
    """This class tests the `tune validate` command."""

    VALID_CONFIG_PATH = ASSETS / "valid_dummy_config.yaml"
    INVALID_CONFIG_PATH = ASSETS / "invalid_dummy_config.yaml"

    def test_validate_good_config(self, capsys, monkeypatch):
        args = f"tune validate {self.VALID_CONFIG_PATH}".split()

        monkeypatch.setattr(sys, "argv", args)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        out = captured.out.rstrip("\n")

        assert out == "Config is well-formed!"

    def test_validate_bad_config(self, monkeypatch, capsys):
        args = f"tune validate {self.INVALID_CONFIG_PATH}".split()

        monkeypatch.setattr(sys, "argv", args)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        err = captured.err.rstrip("\n")

        assert "got an unexpected keyword argument 'dummy'" in err
