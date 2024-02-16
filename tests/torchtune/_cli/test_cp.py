#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys
import tempfile
from pathlib import Path

import pytest

from tests._scripts.common import TUNE_PATH


class TestTuneCLIWithCopyScript:
    def test_copy_successful(self, capsys, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = "tune cp alpaca_llama2_full_finetune.yaml .".split()

            monkeypatch.setattr(sys, "argv", args)
            runpy.run_path(TUNE_PATH, run_name="__main__")

            captured = capsys.readouterr()
            output = captured.err.rstrip("\n")

            assert output == "", f"Expected no output, got {output}"

    def test_copy_successful_when_dest_already_exists(self, capsys, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            existing_file = tmpdir_path / "existing_file.yaml"
            existing_file.touch()

            args = f"tune cp alpaca_llama2_full_finetune.yaml {existing_file}".split()

            monkeypatch.setattr(sys, "argv", args)
            runpy.run_path(TUNE_PATH, run_name="__main__")

            captured = capsys.readouterr()
            output = captured.err.rstrip("\n")

            assert output == "", f"Expected no output, got {output}"

    def test_copy_fails_when_dest_already_exists_and_no_clobber_is_true(
        self, capsys, monkeypatch
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            existing_file = tmpdir_path / "existing_file.yaml"
            existing_file.touch()

            args = f"tune cp alpaca_llama2_full_finetune.yaml {existing_file} --no-clobber".split()

            monkeypatch.setattr(sys, "argv", args)
            runpy.run_path(TUNE_PATH, run_name="__main__")

            captured = capsys.readouterr()
            output = captured.out.rstrip("\n")
            err = captured.err.rstrip("\n")

            assert err == "", f"Expected no error output, got {err}"
            assert (
                "not overwriting" in output
            ), f"Expected 'not overwriting' message, got '{output}'"

    def test_copy_fails_when_given_invalid_recipe(self, capsys, monkeypatch):
        args = "tune cp non_existent_recipe .".split()

        monkeypatch.setattr(sys, "argv", args)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.err.rstrip("\n")

        assert (
            "error: Invalid file name: non_existent_recipe. Try 'tune ls' to see all available files to copy."
            in output
        ), f"Expected error message, got {output}"

    def test_copy_fails_when_given_invalid_config(self, capsys, monkeypatch):
        args = "tune cp non_existent_config.yaml .".split()

        monkeypatch.setattr(sys, "argv", args)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.err.rstrip("\n")

        assert (
            "error: Invalid file name: non_existent_config.yaml. Try 'tune ls' to see all available files to copy."
            in output
        ), f"Expected error message, got {output}"

    def test_copy_fails_when_copying_to_invalid_path(self, capsys, monkeypatch):
        args = "tune cp full_finetune /home/mr_bean/full_finetune.py".split()

        monkeypatch.setattr(sys, "argv", args)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.err.rstrip("\n")

        assert (
            "error: Cannot create regular file: '/home/mr_bean/full_finetune.py'. No such file or directory"
            in output
        ), f"Expected error message, got {output}"

    def test_copy_fails_when_no_arguments_given(self, capsys, monkeypatch):
        args = "tune cp".split()

        monkeypatch.setattr(sys, "argv", args)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.err.rstrip("\n")
        assert (
            "error: the following arguments are required: file, destination" in output
        ), f"Expected error message, got {output}"
