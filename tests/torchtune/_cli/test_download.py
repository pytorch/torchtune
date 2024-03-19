#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys
import tempfile
from unittest.mock import patch

import pytest

from tests.common import TUNE_PATH


class TestTuneCLIWithDownloadScript:
    def test_download_no_hf_token_set_for_gated_model(self, capsys):
        model = "meta-llama/Llama-2-7b"
        testargs = f"tune download --repo-id {model}".split()
        with patch.object(sys, "argv", testargs):
            with pytest.raises(ValueError) as e:
                runpy.run_path(TUNE_PATH, run_name="__main__")

    def test_download_calls_snapshot(self, capsys):
        model = "meta-llama/Llama-2-7b"
        with tempfile.TemporaryDirectory() as tmpdir:
            testargs = f"tune download --repo-id {model} --output-dir {tmpdir} --hf-token ABCDEF".split()
            with patch.object(sys, "argv", testargs):
                with patch("huggingface_hub.snapshot_download") as snapshot:
                    runpy.run_path(TUNE_PATH, run_name="__main__")
                    snapshot.assert_called_once()
