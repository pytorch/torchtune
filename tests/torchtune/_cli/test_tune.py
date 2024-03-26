#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import runpy
import sys
import tempfile
from pathlib import Path

import pytest
import torch

import torchtune

from tests.common import TUNE_PATH
from tests.test_utils import assert_expected
from tests.torchtune.models.llama2.scripts.compare_decoder import Transformer

from torchtune import get_all_recipes
from torchtune.models.llama2 import llama2

ASSETS = Path(__file__).parent.parent.parent / "assets"


class TestTuneCLI:
    pass


class TestTuneListCommand:
    """This class tests the `tune ls` command."""

    def test_ls_lists_all_recipes_and_configs(self, capsys, monkeypatch):
        testargs = "tune ls".split()

        monkeypatch.setattr(sys, "argv", testargs)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        output = captured.out.rstrip("\n")

        for recipe in get_all_recipes():
            assert recipe.uuid in output
            for config in recipe.get_configs():
                assert config.uuid in output


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


class TestTuneCLIWithConvertCheckpointScript:
    # Generating `tiny_state_dict_with_one_key.pt`
    # >>> import torch
    # >>> state_dict = {"test_key": torch.randn(10, 10)}
    # >>> torch.save(state_dict, "tiny_state_dict_with_one_key.pt")

    # Generating `tiny_fair_checkpoint.pt`
    # >>> from tests.torchtune.models.llama2.scripts.compare_decoder import Transformer
    # >>> from tests.test_utils import init_weights_with_constant
    # >>> import torch
    # >>> tiny_fair_transfomer = Transformer(
    #     vocab_size=500,
    #     n_layers=2,
    #     n_heads=4,
    #     dim=32,
    #     max_seq_len=64,
    #     n_kv_heads=4,
    # )
    # >>> init_weights_with_constant(tiny_fair_transfomer, constant=0.2)
    # >>> torch.save(tiny_fair_transfomer.state_dict(), "tiny_fair_checkpoint.pt")

    def test_convert_checkpoint_errors_on_bad_conversion(self, capsys, monkeypatch):
        incorrect_state_dict_loc = ASSETS / "tiny_state_dict_with_one_key.pt"
        testargs = (
            f"tune convert_checkpoint {incorrect_state_dict_loc} --model llama2 --train-type full"
        ).split()
        monkeypatch.setattr(sys, "argv", testargs)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        captured = capsys.readouterr()
        out = captured.out.rstrip("\n")
        err = captured.err.rstrip("\n")

        assert "Error converting the original Llama2" in err

    def _tiny_fair_transformer(self, ckpt):
        tiny_fair_transfomer = Transformer(
            vocab_size=500,
            n_layers=2,
            n_heads=4,
            dim=32,
            max_seq_len=64,
            n_kv_heads=4,
        )
        tiny_fair_state_dict = torch.load(ckpt, weights_only=True)
        tiny_fair_transfomer.load_state_dict(tiny_fair_state_dict, strict=True)
        return tiny_fair_transfomer

    def _tiny_native_transformer(self, ckpt):
        tiny_native_transfomer = llama2(
            vocab_size=500,
            num_layers=2,
            num_heads=4,
            embed_dim=32,
            max_seq_len=64,
            num_kv_heads=4,
        )
        tiny_native_state_dict = torch.load(ckpt, weights_only=True)
        tiny_native_transfomer.load_state_dict(
            tiny_native_state_dict["model"], strict=False
        )
        return tiny_native_transfomer

    def _llama2_7b_fair_transformer(self, ckpt):
        llama2_7b_fair_transformer = Transformer(
            vocab_size=32_000,
            n_layers=32,
            n_heads=32,
            dim=4096,
            max_seq_len=2048,
            n_kv_heads=32,
        )
        llama2_7b_fair_state_dict = torch.load(ckpt, weights_only=True)
        llama2_7b_fair_transformer.load_state_dict(
            llama2_7b_fair_state_dict, strict=False
        )
        llama2_7b_fair_transformer.eval()
        return llama2_7b_fair_transformer

    def _llama2_7b_native_transformer(self, ckpt):
        llama2_7b_native_transformer = llama2(
            vocab_size=32_000,
            num_layers=32,
            num_heads=32,
            embed_dim=4096,
            max_seq_len=2048,
            num_kv_heads=32,
        )
        llama2_7b_native_state_dict = torch.load(ckpt, weights_only=True)
        llama2_7b_native_transformer.load_state_dict(
            llama2_7b_native_state_dict["model"], strict=True
        )
        llama2_7b_native_transformer.eval()
        return llama2_7b_native_transformer

    def _generate_toks_for_tiny(self):
        return torch.randint(low=0, high=500, size=(16, 64))

    def _generate_toks_for_llama2_7b(self):
        return torch.randint(low=0, high=32_000, size=(16, 128))

    def test_convert_checkpoint_matches_fair_model(
        self, capsys, pytestconfig, monkeypatch
    ):
        is_large_scale_test = pytestconfig.getoption("--large-scale")

        if is_large_scale_test:
            ckpt = "/tmp/test-artifacts/llama2-7b-fair"
            fair_transformer = self._llama2_7b_fair_transformer(ckpt)
        else:
            ckpt = ASSETS / "tiny_fair_checkpoint.pt"
            fair_transformer = self._tiny_fair_transformer(ckpt)

        output_path = tempfile.NamedTemporaryFile(delete=True).name
        testargs = (
            f"tune convert_checkpoint {ckpt} --output-path {output_path} --model llama2 --train-type lora"
        ).split()
        monkeypatch.setattr(sys, "argv", testargs)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        output = capsys.readouterr().out.rstrip("\n")
        assert "Succesfully wrote PyTorch-native model checkpoint" in output

        native_transformer = (
            self._llama2_7b_native_transformer(output_path)
            if is_large_scale_test
            else self._tiny_native_transformer(output_path)
        )

        with torch.no_grad():
            for i in range(10):
                toks = (
                    self._generate_toks_for_llama2_7b()
                    if is_large_scale_test
                    else self._generate_toks_for_tiny()
                )
                fair_out = fair_transformer(toks)
                native_out = native_transformer(toks)
                assert_expected(fair_out.sum(), native_out.sum())
