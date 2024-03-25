#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys

import pytest

from tests.common import TUNE_PATH


class TestTuneCLIWithDownloadScript:
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
