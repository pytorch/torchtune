# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import runpy
import sys
from pathlib import Path

import pytest

import torchtune
from tests.common import TUNE_PATH


pkg_path = Path(torchtune.__file__).parent.absolute()
EVAL_CONFIG_PATH = Path.joinpath(
    pkg_path, "_cli", "eval_configs", "default_eval_config.yaml"
)
from tests.recipes.utils import llama2_test_config
from tests.test_utils import CKPT_MODEL_PATHS

# TODO: Move this to tests/recipes once we convert eval script into a recipe
class TestEval:
    def test_torchune_checkpoint_eval_result(self, capsys, monkeypatch):
        ckpt = "small_test_ckpt_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        cpu_device_str = "cpu"
        tokenizer_pth = "/tmp/test-artifacts/tokenizer.model"

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        cmd = f"""
        tune eval \
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
