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

from tests.common import TUNE_PATH

from torchtune import get_all_recipes


class TestTuneRunCommand:
    def test_run_calls_distributed_run_for_distributed_recipe(
        self, capsys, monkeypatch, mocker
    ):
        testargs = "tune run --num-gpu 4 full_finetune_distributed --config llama2/7B_full".split()

        monkeypatch.setattr(sys, "argv", testargs)
        distributed_run = mocker.patch("torch.distributed.run.run")
        runpy.run_path(TUNE_PATH, run_name="__main__")
        distributed_run.assert_called_once()

        output = capsys.readouterr()
        assert "Running with torchrun..." in output.out

    def test_run_calls_single_device_run_for_single_device_recipe(
        self, capsys, monkeypatch, mocker
    ):
        testargs = "tune run full_finetune_single_device --config llama2/7B_full_single_device".split()

        monkeypatch.setattr(sys, "argv", testargs)
        single_device_run = mocker.patch.object(
            torchtune._cli.tune.Run, "_run_single_device", autospec=True
        )
        runpy.run_path(TUNE_PATH, run_name="__main__")
        single_device_run.assert_called_once()

    def test_run_fails_when_called_with_distributed_args_for_single_device_recipe(
        self, capsys, monkeypatch
    ):
        testargs = "tune run --num-gpu 4 full_finetune_single_device --config llama2/7B_full_single_device".split()

        monkeypatch.setattr(sys, "argv", testargs)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        output = capsys.readouterr()
        assert "does not support distributed training" in output.err

    def test_run_calls_local_file_run_for_local_file_recipe(
        self, capsys, monkeypatch, mocker
    ):
        testargs = "tune run my_custom_recipe.py --config custom_config.yaml".split()

        monkeypatch.setattr(sys, "argv", testargs)
        local_file_run = mocker.patch("torchtune._cli.tune.Run._run_single_device")
        runpy.run_path(TUNE_PATH, run_name="__main__")
        local_file_run.assert_called_once()

    def test_run_fails_when_using_custom_recipe_and_default_config(
        self, capsys, monkeypatch
    ):
        testargs = "tune run my_custom_recipe.py --config llama2/7B_full".split()

        monkeypatch.setattr(sys, "argv", testargs)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        output = capsys.readouterr()
        assert "please copy the config file to your local dir first" in output.err
