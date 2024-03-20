# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import runpy
import sys

import pytest

from tests.common import TUNE_PATH
from tests.recipes.common import RECIPE_TESTS_DIR
from tests.recipes.utils import fetch_ckpt_model_path, llama2_test_config

_CONFIG_PATH = RECIPE_TESTS_DIR / "alpaca_generate_test_config.yaml"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAlpacaGenerateRecipe:
    def test_alpaca_generate(self, capsys, pytestconfig, tmpdir, monkeypatch):
        large_scale = pytestconfig.getoption("--large-scale")
        ckpt = "llama2.llama2_7b" if large_scale else "small_test_ckpt_tune"

        cmd = f"""
        tune alpaca_generate
            --config alpaca_llama2_generate \
            model_checkpoint={fetch_ckpt_model_path(ckpt)} \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
            output_dir={tmpdir} \
        """.split()

        model_config = (
            ["model=torchtune.models.llama2.llama2_7b"]
            if large_scale
            else llama2_test_config()
        )
        cmd += model_config

        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")
