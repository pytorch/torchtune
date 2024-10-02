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
from tests.recipes.utils import MODEL_TEST_CONFIGS, write_hf_ckpt_config
from tests.test_utils import CKPT_MODEL_PATHS, mps_ignored_test, TOKENIZER_PATHS


class TestGenerateV2:
    """Recipe test suite for the generate_v2 recipe."""

    @pytest.mark.integration_test
    @mps_ignored_test()
    def test_llama2_generate_results(self, caplog, monkeypatch, tmpdir):
        ckpt = "llama2_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS["llama2"])
        ckpt_dir = ckpt_path.parent

        # Config file needed for model conversion.
        write_hf_ckpt_config(ckpt_dir)

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
            device=cpu \
            dtype=fp32 \
            max_new_tokens=10 \
            seed=123 \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama2"]
        cmd = cmd + model_config

        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # this is gibberish b/c the model is random weights, but it's
        # the expected value for what we currently have in V2
        # this test should catch any changes to the generate recipe that affect output
        expected_output = (
            "Country maior Connection Kohćutsójcustomulas Sometimes Security"
        )

        logs = caplog.text
        assert expected_output in logs

    @pytest.mark.integration_test
    def test_llama2_fail_on_bad_input(self, capsys, monkeypatch, tmpdir):
        """Should fail when user passes in a bad input:
        - No prompt provided
        - Prompt has multiple entries in content and no image
        """
        pass
