# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import math
import re
import runpy
import sys
from pathlib import Path

import pytest

from tests.common import TUNE_PATH
from tests.recipes.utils import llama2_test_config
from tests.test_utils import CKPT_MODEL_PATHS


class TestGenerateV2:
    """Recipe test suite for the generate_v2 recipe."""

    @pytest.mark.integration_test
    def test_llama2_generate_results(
        self, capsys, monkeypatch, tmpdir, prompt
    ):
        ckpt = "llama2_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent

        cmd = f"""
        tune run dev/generate_v2 \
            --config llama2/generation_v2 \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
        """.split()

        model_config = llama2_test_config()
        cmd = cmd + model_config

        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_output = "Of course! The capital of France is Paris. 🇫🇷"

        out = capsys.readouterr().out
        assert expected_output in out
    
