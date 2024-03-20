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


class TestTuneCLIWithCopyScript:
    @pytest.mark.parametrize("already_exists", (True, False))
    def test_copy_successful(self, capsys, monkeypatch, tmpdir, already_exists):
        tmpdir_path = Path(tmpdir)
        dest = tmpdir_path / "my_custom_finetune.yaml"

        if already_exists:
            dest.touch()

        args = f"tune cp full_finetune_single_device.yaml {dest}".split()

        monkeypatch.setattr(sys, "argv", args)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        out = captured.out.rstrip("\n")

        assert dest.exists(), f"Expected {dest} to exist"
        assert out == ""

    def test_copy_skips_when_dest_already_exists_and_no_clobber_is_true(
        self, capsys, monkeypatch, tmpdir
    ):
        tmpdir_path = Path(tmpdir)
        existing_file = tmpdir_path / "existing_file.yaml"
        existing_file.touch()

        args = f"tune cp full_finetune_single_device.yaml {existing_file} -n".split()

        monkeypatch.setattr(sys, "argv", args)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        out = captured.out.rstrip("\n")
        err = captured.err.rstrip("\n")

        assert err == ""
        assert (
            "not overwriting" in out
        ), f"Expected 'not overwriting' message, got '{out}'"

    @pytest.mark.parametrize(
        "tune_command,expected_error_message",
        [
            (
                "tune cp non_existent_recipe.py .",
                "error: Invalid file name: non_existent_recipe.py. Try `tune ls` to see all available files to copy.",
            ),
            (
                "tune cp non_existent_config.yaml .",
                "error: Invalid file name: non_existent_config.yaml. Try `tune ls` to see all available files to copy.",
            ),
            (
                "tune cp full_finetune_single_device.py /home/mr_bean/full_finetune_single_device.py",
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
