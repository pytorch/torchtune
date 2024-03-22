# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import runpy

import sys
from pathlib import Path

import pytest

import torchtune
from tests.common import TUNE_PATH
from tests.recipes.utils import fetch_ckpt_model_path, llama2_small_test_ckpt
from torchtune import models

pkg_path = Path(torchtune.__file__).parent.parent.absolute()
EVAL_CONFIG_PATH = Path.joinpath(
    pkg_path, "recipes", "configs", "llama2_eleuther_eval.yaml"
)

models.small_test_ckpt_tune = llama2_small_test_ckpt


class TestEleutherEval:
    tokenizer_pth = "/tmp/test-artifacts/tokenizer.model"
    model_ckpt = "small_test_ckpt_tune"

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def test_torchune_checkpoint_eval_result(self, caplog, monkeypatch, tmpdir):
        ckpt_path = Path(fetch_ckpt_model_path(self.model_ckpt))
        ckpt_dir = ckpt_path.parent

        cmd = f"""
        tune eleuther_eval \
            --config {EVAL_CONFIG_PATH} \
            model._component_=torchtune.models.{self.model_ckpt} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            tokenizer._component_=torchtune.models.llama2.llama2_tokenizer \
            tokenizer.path={self.tokenizer_pth} \
            limit=10 \
            dtype=fp32 \
            device=cpu \
        """.split()

        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        log_out = caplog.messages[-1]
        assert "'acc,none': 0.3" in log_out

    @pytest.fixture
    def hide_available_pkg(self, monkeypatch):
        import_orig = builtins.__import__

        def mocked_import(name, *args, **kwargs):
            if name == "lm_eval":
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)

    @pytest.mark.usefixtures("hide_available_pkg")
    def test_eval_recipe_errors_without_lm_eval(self, caplog, monkeypatch, tmpdir):
        ckpt_path = Path(fetch_ckpt_model_path(self.model_ckpt))
        ckpt_dir = ckpt_path.parent

        cmd = f"""
        tune eleuther_eval \
            --config {EVAL_CONFIG_PATH} \
            model._component_=torchtune.models.{self.model_ckpt} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            tokenizer._component_=torchtune.models.llama2.llama2_tokenizer \
            tokenizer.path={self.tokenizer_pth} \
            limit=10 \
            dtype=fp32 \
            device=cpu \
        """.split()

        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        log_out = caplog.messages[0]
        assert "Recipe requires EleutherAI Eval Harness v0.4" in log_out
