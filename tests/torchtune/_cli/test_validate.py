# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys

import pytest

from tests.common import TUNE_PATH
from torchtune.config._errors import ConfigError

VALID_CONFIG_PATH = "tests/assets/valid_dummy_config.yaml"
INVALID_CONFIG_PATH = "tests/assets/invalid_dummy_config.yaml"


class TestTuneCLIWithValidateScript:
    def test_validate_good_config(self, capsys, monkeypatch):
        args = f"tune validate --config {VALID_CONFIG_PATH}".split()

        monkeypatch.setattr(sys, "argv", args)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        out = captured.out.rstrip("\n")

        assert out == "Config is well-formed!"

    def test_validate_bad_config(self, monkeypatch):
        args = f"tune validate --config {INVALID_CONFIG_PATH}".split()

        monkeypatch.setattr(sys, "argv", args)
        with pytest.raises(
            ConfigError, match="got an unexpected keyword argument 'dummy'"
        ):
            runpy.run_path(TUNE_PATH, run_name="__main__")

    def test_validate_bad_override(self, monkeypatch, tmpdir):
        args = f"\
            tune validate --config {VALID_CONFIG_PATH} \
            test._component_=torchtune.utils.get_dtype \
            test.dtype=fp32 test.dummy=3".split()

        monkeypatch.setattr(sys, "argv", args)
        with pytest.raises(
            ConfigError, match="got an unexpected keyword argument 'dummy'"
        ):
            runpy.run_path(TUNE_PATH, run_name="__main__")
