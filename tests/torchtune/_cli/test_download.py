import runpy
import sys

import pytest
import huggingface_hub
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from tests.common import TUNE_PATH


class TestTuneDownloadCommand:
    """This class tests the `tune download` command."""

    @pytest.fixture
    def model_info(self, mocker):
        return mocker.patch.object(
            huggingface_hub,
            "model_info",
            return_value=None,
            # Side effects are iterated through on each call
            side_effect=[GatedRepoError("test"), RepositoryNotFoundError("test"), mocker.DEFAULT],
        )

    def test_download_calls_model_info_error(self, capsys, monkeypatch, model_info):
        model = "meta-llama/Llama-2-7b"
        testargs = f"tune download {model}".split()
        monkeypatch.setattr(sys, "argv", testargs)

        # Call the first time and get GatedRepoError
        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(TUNE_PATH, run_name="__main__")
        err = capsys.readouterr().err
        assert (
            "You need to provide a Hugging Face API token to download gated models"
            in err
        )

        # Call the second time and get RepositoryNotFoundError
        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(TUNE_PATH, run_name="__main__")
        err = capsys.readouterr().err
        assert "not found on the HuggingFace Hub" in err

        # Make sure it was called twice
        assert model_info.call_count == 2

    def test_download_calls_snapshot(self, capsys, tmpdir, monkeypatch, model_info, mocker):
        model = "mistralai/Mistral-7b-v0.1"
        testargs = f"tune download {model} --output-dir {tmpdir}".split()

        snapshot_download = mocker.patch.object(
            huggingface_hub,
            "snapshot_download",
            return_value=tmpdir,
        )

        monkeypatch.setattr(sys, "argv", testargs)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        snapshot_download.assert_called_once()
