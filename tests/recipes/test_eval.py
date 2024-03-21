# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys
from pathlib import Path

import pytest

import torchtune
from tests.common import TUNE_PATH

models.small_test_ckpt_tune = llama2_small_test_ckpt


class TestEval:
    @pytest.mark.integration_test
    def test_torchune_checkpoint_eval_result(self, capsys, monkeypatch):
        ckpt = "small_test_ckpt_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        cpu_device_str = "cpu"
        tokenizer_pth = "/tmp/test-artifacts/tokenizer.model"

        cmd = f"""
        tune eleuther_eval \
            --config {EVAL_CONFIG_PATH} \
            model_checkpoint={ckpt_path} \
            tokenizer._component_=torchtune.models.llama2.llama2_tokenizer \
            tokenizer.path={tokenizer_pth} \
            limit=10 \
            device={cpu_device_str} \
        """.split()
        cmd = cmd + llama2_test_config()
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        out_err = capsys.readouterr()
        assert "'acc,none': 0.3" in out_err.out

    def test_eval_recipe_errors_without_lm_eval(self, capsys, monkeypatch):
        cmd = f"""
        tune eleuther_eval \
            --config {EVAL_CONFIG_PATH} \
            model._component_=torchtune.models.llama2.llama2_tokenizer \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
            limit=10 \
        """.split()
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        out_err = capsys.readouterr()
        assert "Missing required argument" in out_err.err
