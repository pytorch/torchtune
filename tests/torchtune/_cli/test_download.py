# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys

import huggingface_hub

import pytest
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from tests.common import TUNE_PATH


class TestTuneDownloadCommand:
    """This class tests the `tune download` command."""

    @pytest.fixture
    def snapshot_download(self, mocker, tmpdir):
        return mocker.patch.object(
            huggingface_hub,
            "snapshot_download",
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
        err = capsys.readouterr().err
        assert (
            "You need to provide a HuggingFace API token to download gated models"
            in err
        )

        # Call the second time and get RepositoryNotFoundError
        with pytest.raises(SystemExit, match="2"):
            runpy.run_path(TUNE_PATH, run_name="__main__")
        err = capsys.readouterr().err
        assert "not found on the HuggingFace Hub" in err

        # Call the third time and get the expected output
        runpy.run_path(TUNE_PATH, run_name="__main__")
        output = capsys.readouterr().out
        assert "Succesfully downloaded model repo" in output

        # Make sure it was called twice
        assert snapshot_download.call_count == 3
