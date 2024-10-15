# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys

import pytest
from tests.common import TUNE_PATH


class TestTuneDownloadCommand:
    """This class tests the `tune download` command."""

    @pytest.fixture
    def snapshot_download(self, mocker, tmpdir):

        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

        yield mocker.patch(
            "torchtune._cli.download.snapshot_download",
            return_value=tmpdir,
            # Side effects are iterated through on each call
            side_effect=[
                GatedRepoError("test"),
                RepositoryNotFoundError("test"),
                mocker.DEFAULT,
            ],
        )

    def test_download_calls_snapshot(self, capsys, monkeypatch, snapshot_download):
        model = "meta-llama/Llama-2-7b"
        testargs = f"tune download {model}".split()
        monkeypatch.setattr(sys, "argv", testargs)

        # Call the first time and get GatedRepoError
        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(TUNE_PATH, run_name="__main__")
        out_err = capsys.readouterr()
        assert (
            "Ignoring files matching the following patterns: *.safetensors"
            in out_err.out
        )
        assert (
            "It looks like you are trying to access a gated repository." in out_err.err
        )

        # Call the second time and get RepositoryNotFoundError
        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(TUNE_PATH, run_name="__main__")
        out_err = capsys.readouterr()
        assert (
            "Ignoring files matching the following patterns: *.safetensors"
            in out_err.out
        )
        assert "not found on the Hugging Face Hub" in out_err.err

        # Call the third time and get the expected output
        runpy.run_path(TUNE_PATH, run_name="__main__")
        output = capsys.readouterr().out
        assert "Ignoring files matching the following patterns: *.safetensors" in output
        assert "Successfully downloaded model repo" in output

        # Make sure it was called twice
        assert snapshot_download.call_count == 3

    # GatedRepoError without --hf-token (expect prompt for token)
    def test_gated_repo_error_no_token(self, capsys, monkeypatch, snapshot_download):
        model = "meta-llama/Llama-2-7b"
        testargs = f"tune download {model}".split()
        monkeypatch.setattr(sys, "argv", testargs)

        # Expect GatedRepoError without --hf-token provided
        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        out_err = capsys.readouterr()
        # Check that error message prompts for --hf-token
        assert (
            "It looks like you are trying to access a gated repository." in out_err.err
        )
        assert (
            "Please ensure you have access to the repository and have provided the proper Hugging Face API token"
            in out_err.err
        )

    # GatedRepoError with --hf-token (should not ask for token)
    def test_gated_repo_error_with_token(self, capsys, monkeypatch, snapshot_download):
        model = "meta-llama/Llama-2-7b"
        testargs = f"tune download {model} --hf-token valid_token".split()
        monkeypatch.setattr(sys, "argv", testargs)

        # Expect GatedRepoError with --hf-token provided
        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        out_err = capsys.readouterr()
        # Check that error message does not prompt for --hf-token again
        assert (
            "It looks like you are trying to access a gated repository." in out_err.err
        )
        assert "Please ensure you have access to the repository." in out_err.err
        assert (
            "Please ensure you have access to the repository and have provided the proper Hugging Face API token"
            not in out_err.err
        )
