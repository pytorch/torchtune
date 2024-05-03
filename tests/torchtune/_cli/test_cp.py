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

    def test_copy_successful_with_cwd_as_path(self, capsys, monkeypatch, tmpdir):
        tmpdir_path = Path(tmpdir)

        # Needed so we can run test from tmpdir
        tune_path_as_absolute = Path(TUNE_PATH).absolute()

        # Change cwd to tmpdir
        monkeypatch.chdir(tmpdir_path)

        args = "tune cp llama2/7B_full .".split()
        monkeypatch.setattr(sys, "argv", args)
        runpy.run_path(str(tune_path_as_absolute), run_name="__main__")

        captured = capsys.readouterr()
        out = captured.out.rstrip("\n")

        dest = tmpdir_path / "7B_full.yaml"

        assert dest.exists()
        assert "Copied file to ./7B_full.yaml" in out

    def test_copy_skips_when_dest_already_exists_and_no_clobber_is_true(
        self, capsys, monkeypatch, tmpdir
    ):
        tmpdir_path = Path(tmpdir)
        existing_file = tmpdir_path / "existing_file.yaml"
        existing_file.touch()

        args = f"tune cp llama2/7B_full_low_memory {existing_file} -n".split()

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

        args = f"tune cp llama2/7B_full_low_memory {dest}".split()

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
