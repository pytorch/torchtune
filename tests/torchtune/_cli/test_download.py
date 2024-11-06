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

    # only valid --source parameters supported (expect prompt for supported values)
    def test_source_parameter(self, capsys, monkeypatch):
        model = "metaresearch/llama-3.2/pytorch/1b"
        testargs = f"tune download {model} --source invalid".split()
        monkeypatch.setattr(sys, "argv", testargs)

        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        output = capsys.readouterr()
        assert "argument --source: invalid choice: 'invalid'" in output.err

    def test_download_from_kaggle(self, capsys, monkeypatch, mocker, tmpdir):
        model = "metaresearch/llama-3.2/pytorch/1b"
        testargs = f"tune download {model} --source kaggle --kaggle-username kaggle_user --kaggle-api-key kaggle_api_key".split()
        monkeypatch.setattr(sys, "argv", testargs)
        # mock out kagglehub.model_download to get around key storage
        mocker.patch("torchtune._cli.download.model_download", return_value=tmpdir)

        runpy.run_path(TUNE_PATH, run_name="__main__")

        output = capsys.readouterr().out
        assert "Successfully downloaded model repo" in output

    def test_download_from_kaggle_warn_when_output_dir_provided(
        self, capsys, monkeypatch, mocker, tmpdir
    ):
        model = "metaresearch/llama-3.2/pytorch/1b"
        testargs = f"tune download {model} --source kaggle --output-dir /requested/model/path".split()
        monkeypatch.setattr(sys, "argv", testargs)
        # mock out kagglehub.model_download to get around key storage
        mocker.patch("torchtune._cli.download.model_download", return_value=tmpdir)

        with pytest.warns(
            UserWarning,
            match="--output-dir flag is not supported for Kaggle model downloads",
        ):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        output = capsys.readouterr().out
        assert "Successfully downloaded model repo" in output

    # passes partial credentials with just --kaggle-api-key (expect prompt for all necessary credentials)
    def test_download_from_kaggle_partial_credentials_provided(
        self, capsys, monkeypatch
    ):
        model = "metaresearch/llama-3.2/pytorch/1b"
        testargs = (
            f"tune download {model} --source kaggle --kaggle-api-key apikey".split()
        )
        monkeypatch.setattr(sys, "argv", testargs)

        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        out_err = capsys.readouterr()
        assert "Missing --kaggle-username." in out_err.err
        assert "Please provide both your Kaggle username and API key." in out_err.err

        testargs = (
            f"tune download {model} --source kaggle --kaggle-username username".split()
        )
        monkeypatch.setattr(sys, "argv", testargs)

        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        out_err = capsys.readouterr()
        assert "Missing --kaggle-api-key." in out_err.err
        assert "Please provide both your Kaggle username and API key." in out_err.err
        assert "Find your API key at https://kaggle.com/settings." in out_err.err

    # KaggleApiHTTPError::Unauthorized without --kaggle-username and --kaggle-api-key (expect prompt for credentials)
    def test_download_from_kaggle_unauthorized_credentials(
        self, capsys, monkeypatch, mocker
    ):
        from http import HTTPStatus

        from kagglehub.exceptions import KaggleApiHTTPError

        model = "metaresearch/llama-3.2/pytorch/1b"
        testargs = f"tune download {model} --source kaggle --kaggle-username username --kaggle-api-key key".split()
        monkeypatch.setattr(sys, "argv", testargs)

        mock_model_download = mocker.patch("torchtune._cli.download.model_download")
        mock_model_download.side_effect = KaggleApiHTTPError(
            "Unauthorized",
            response=mocker.MagicMock(status_code=HTTPStatus.UNAUTHORIZED),
        )

        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        out_err = capsys.readouterr()
        assert (
            "Please ensure you have access to the model and have provided the proper Kaggle credentials"
            in out_err.err
        )
        assert "You can also set these to environment variables" in out_err.err

    # KaggleApiHTTPError::NotFound
    def test_download_from_kaggle_model_not_found(self, capsys, monkeypatch, mocker):
        from http import HTTPStatus

        from kagglehub.exceptions import KaggleApiHTTPError

        model = "mockorganizations/mockmodel/pytorch/mockvariation"
        testargs = f"tune download {model} --source kaggle --kaggle-username kaggle_user --kaggle-api-key kaggle_api_key".split()
        monkeypatch.setattr(sys, "argv", testargs)

        mock_model_download = mocker.patch("torchtune._cli.download.model_download")
        mock_model_download.side_effect = KaggleApiHTTPError(
            "NotFound", response=mocker.MagicMock(status_code=HTTPStatus.NOT_FOUND)
        )

        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        out_err = capsys.readouterr()
        assert f"'{model}' not found on the Kaggle Model Hub." in out_err.err

    # KaggleApiHTTPError::InternalServerError
    def test_download_from_kaggle_api_error(self, capsys, monkeypatch, mocker):
        from http import HTTPStatus

        from kagglehub.exceptions import KaggleApiHTTPError

        model = "metaresearch/llama-3.2/pytorch/1b"
        testargs = f"tune download {model} --source kaggle --kaggle-username kaggle_user --kaggle-api-key kaggle_api_key".split()
        monkeypatch.setattr(sys, "argv", testargs)

        mock_model_download = mocker.patch("torchtune._cli.download.model_download")
        mock_model_download.side_effect = KaggleApiHTTPError(
            "InternalError",
            response=mocker.MagicMock(status_code=HTTPStatus.INTERNAL_SERVER_ERROR),
        )

        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        out_err = capsys.readouterr()
        assert "Failed to download" in out_err.err

    def test_download_from_kaggle_warn_on_nonmeta_pytorch_models(
        self, monkeypatch, mocker, tmpdir
    ):
        model = "kaggle/kaggle-model-name/pytorch/1b"
        testargs = f"tune download {model} --source kaggle".split()
        monkeypatch.setattr(sys, "argv", testargs)

        # stub out model_download to guarantee success
        mocker.patch(
            "torchtune._cli.download.model_download",
            return_value=tmpdir,
        )

        with pytest.warns(UserWarning, match="may not be compatible with torchtune"):
            runpy.run_path(TUNE_PATH, run_name="__main__")

    def test_download_from_kaggle_warn_on_nonpytorch_nontransformers_model(
        self, monkeypatch, mocker, tmpdir
    ):
        model = "metaresearch/some-model/some-madeup-framework/1b"
        testargs = f"tune download {model} --source kaggle".split()
        monkeypatch.setattr(sys, "argv", testargs)

        # stub out model_download to guarantee success
        mocker.patch(
            "torchtune._cli.download.model_download",
            return_value=tmpdir,
        )

        with pytest.warns(UserWarning, match="may not be compatible with torchtune"):
            runpy.run_path(TUNE_PATH, run_name="__main__")
