# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys

import pytest

from tests.common import TUNE_PATH


class TestTuneRunCommand:
    def test_run_calls_distributed_run_for_distributed_recipe(
        self, capsys, monkeypatch, mocker
    ):
        testargs = "tune run --nproc_per_node 4 full_finetune_distributed --config llama2/7B_full".split()

        monkeypatch.setattr(sys, "argv", testargs)
        distributed_run = mocker.patch("torchtune._cli.tune.Run._run_distributed")
        runpy.run_path(TUNE_PATH, run_name="__main__")
        distributed_run.assert_called_once()

    def test_run_calls_single_device_run_for_single_device_recipe(
        self, capsys, monkeypatch, mocker
    ):
        testargs = "tune run full_finetune_single_device --config llama2/7B_full_single_device".split()

        monkeypatch.setattr(sys, "argv", testargs)
        single_device_run = mocker.patch("torchtune._cli.tune.Run._run_single_device")
        runpy.run_path(TUNE_PATH, run_name="__main__")
        single_device_run.assert_called_once()

    def test_run_fails_when_called_with_distributed_args_for_single_device_recipe(
        self, capsys, monkeypatch
    ):
        testargs = "tune run --nproc_per_node 4 full_finetune_single_device --config llama2/7B_full_single_device".split()

        monkeypatch.setattr(sys, "argv", testargs)
        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        output = capsys.readouterr()
        assert "does not support distributed training" in output.err

    def test_run_fails_when_config_not_passed_in(self, capsys, monkeypatch):
        testargs = "tune run full_finetune_single_device batch_size=3".split()

        monkeypatch.setattr(sys, "argv", testargs)
        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        output = capsys.readouterr()
        assert "The '--config' argument is required" in output.err

    def test_run_succeeds_with_local_recipe_file_and_default_config(
        self, capsys, monkeypatch, mocker
    ):
        testargs = "tune run my_custom_recipe.py --config llama2/7B_full".split()
        monkeypatch.setattr(sys, "argv", testargs)
        local_file_run = mocker.patch("torchtune._cli.tune.Run._run_single_device")
        runpy.run_path(TUNE_PATH, run_name="__main__")
        local_file_run.assert_called_once()

    def test_run_calls_local_file_run_for_local_file_recipe(
        self, capsys, monkeypatch, mocker
    ):
        testargs = "tune run my_custom_recipe.py --config custom_config.yaml".split()

        monkeypatch.setattr(sys, "argv", testargs)
        local_file_run = mocker.patch("torchtune._cli.tune.Run._run_single_device")
        runpy.run_path(TUNE_PATH, run_name="__main__")
        local_file_run.assert_called_once()
