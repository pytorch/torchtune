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

ASSETS = Path(__file__).parent.parent.parent / "assets"


class TestTuneListCommand:
    """This class tests the `tune ls` command."""

    def test_ls_lists_all_recipes_and_configs(self, capsys, monkeypatch):
        testargs = "tune ls".split()

        monkeypatch.setattr(sys, "argv", testargs)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")

        for recipe in get_all_recipes():
            assert recipe.name in output
            for config in recipe.get_configs():
                assert config.name in output


class TestTuneCLIWithCopyScript:
    """This class tests the `tune cp` command."""

    @pytest.mark.parametrize("already_exists", (True, False))
    def test_copy_successful(self, capsys, monkeypatch, tmpdir, already_exists):
        tmpdir_path = Path(tmpdir)
        dest = tmpdir_path / "my_custom_finetune.yaml"

        if already_exists:
            dest.touch()

        args = f"tune cp llama2/7B_full {dest}".split()

        monkeypatch.setattr(sys, "argv", args)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        out = captured.out.rstrip("\n")

        assert dest.exists(), f"Expected {dest} to exist"
        assert f"Copied file to {dest}" in out

    def test_copy_skips_when_dest_already_exists_and_no_clobber_is_true(
        self, capsys, monkeypatch, tmpdir
    ):
        tmpdir_path = Path(tmpdir)
        existing_file = tmpdir_path / "existing_file.yaml"
        existing_file.touch()

        args = f"tune cp llama2/7B_full_single_device {existing_file} -n".split()

        monkeypatch.setattr(sys, "argv", args)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        out = captured.out.rstrip("\n")
        err = captured.err.rstrip("\n")

        assert err == ""
        assert "not overwriting" in out

    def test_adds_correct_suffix_to_dest_when_no_suffix_is_provided(
        self, capsys, monkeypatch, tmpdir
    ):
        tmpdir_path = Path(tmpdir)
        dest = tmpdir_path / "my_custom_finetune"

        args = f"tune cp llama2/7B_full_single_device {dest}".split()

        monkeypatch.setattr(sys, "argv", args)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        out = captured.out.rstrip("\n")

        assert dest.with_suffix(".yaml").exists(), f"Expected {dest} to exist"
        assert f"Copied file to {dest}.yaml" in out

    @pytest.mark.parametrize(
        "tune_command,expected_error_message",
        [
            (
                "tune cp non_existent_recipe .",
                "error: Invalid file name: non_existent_recipe. Try `tune ls` to see all available files to copy.",
            ),
            (
                "tune cp non_existent_config .",
                "error: Invalid file name: non_existent_config. Try `tune ls` to see all available files to copy.",
            ),
            (
                "tune cp full_finetune_single_device /home/mr_bean/full_finetune_single_device.py",
                "error: Cannot create regular file: '/home/mr_bean/full_finetune_single_device.py'. No such file or directory.",
            ),
            (
                "tune cp",
                "error: the following arguments are required: file, destination",
            ),
        ],
    )
    def test_copy_fails_when_given_invalid_recipe(
        self, capsys, monkeypatch, tune_command, expected_error_message
    ):
        args = tune_command.split()

        monkeypatch.setattr(sys, "argv", args)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        err = captured.err.rstrip("\n")

        assert expected_error_message in err


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


class TestTuneDownloadCommand:
    """This class tests the `tune download` command."""

    def test_download_no_hf_token_set_for_gated_model(self, capsys, monkeypatch):
        model = "meta-llama/Llama-2-7b"
        testargs = f"tune download {model}".split()
        monkeypatch.setattr(sys, "argv", testargs)
        with pytest.raises(SystemExit) as e:
            runpy.run_path(TUNE_PATH, run_name="__main__")

    def test_download_calls_snapshot(self, capsys, tmpdir, monkeypatch, mocker):
        model = "meta-llama/Llama-2-7b"
        testargs = (
            f"tune download {model} --output-dir {tmpdir} --hf-token ABCDEF".split()
        )
        monkeypatch.setattr(sys, "argv", testargs)
        with mocker.patch("huggingface_hub.snapshot_download") as snapshot:
            # This error is to be expected b/c we don't actually make the download call
            # in the test. Therefore, there are no files to be found.
            with pytest.raises(FileNotFoundError):
                runpy.run_path(TUNE_PATH, run_name="__main__")
                snapshot.assert_called_once()


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
