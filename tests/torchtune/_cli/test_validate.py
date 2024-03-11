#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tests.common import TUNE_PATH


class TestTuneCLIWithValidateScript:
    def valid_config_string(self):
        return """
        test:
          _component_: torchtune.utils.get_dtype
          dtype: fp32
        """

    def invalid_config_string(self):
        return """
        test:
          _component_: torchtune.utils.get_dtype
          dtype: fp32
          dummy: 3
        """

    def test_validate_good_config(self, capsys, monkeypatch, tmpdir):
        tmpdir_path = Path(tmpdir)
        dest = tmpdir_path / "my_custom_finetune.yaml"
        conf = OmegaConf.create(self.valid_config_string())
        OmegaConf.save(conf, dest)

        args = f"tune validate --config {dest}".split()

        monkeypatch.setattr(sys, "argv", args)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        out = captured.out.rstrip("\n")

        assert out == "Config is well-formed!"

    def test_validate_bad_config(self, monkeypatch, tmpdir):
        tmpdir_path = Path(tmpdir)
        dest = tmpdir_path / "my_custom_finetune.yaml"
        conf = OmegaConf.create(self.invalid_config_string())
        OmegaConf.save(conf, dest)

        args = f"tune validate --config {dest}".split()

        monkeypatch.setattr(sys, "argv", args)
        with pytest.raises(
            TypeError, match="got an unexpected keyword argument 'dummy'"
        ):
            runpy.run_path(TUNE_PATH, run_name="__main__")

    def test_validate_bad_override(self, monkeypatch, tmpdir):
        tmpdir_path = Path(tmpdir)
        dest = tmpdir_path / "my_custom_finetune.yaml"
        conf = OmegaConf.create(self.valid_config_string())
        OmegaConf.save(conf, dest)

        args = f"\
            tune validate --config {dest} --override \
            test._component_=torchtune.utils.get_dtype \
            test.dtype=fp32 test.dummy=3".split()

        monkeypatch.setattr(sys, "argv", args)
        with pytest.raises(
            TypeError, match="got an unexpected keyword argument 'dummy'"
        ):
            runpy.run_path(TUNE_PATH, run_name="__main__")
