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

from tests.recipes.utils import llama2_small_test_ckpt
from torchtune import models

_CONFIG_PATH = RECIPE_TESTS_DIR / "alpaca_generate_test_config.yaml"

models.small_test_ckpt = llama2_small_test_ckpt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAlpacaGenerateRecipe:
    def _fetch_ckpt_model_path(self, ckpt) -> str:
        if ckpt == "small_test_ckpt":
            return "/tmp/test-artifacts/small-ckpt-01242024"
        if ckpt == "llama2.llama2_7b":
            return "/tmp/test-artifacts/llama2-7b-01242024"
        raise ValueError(f"Unknown ckpt {ckpt}")

    def test_alpaca_generate(self, capsys, pytestconfig, tmpdir, monkeypatch):
        large_scale = pytestconfig.getoption("--large-scale")
        ckpt = "llama2.llama2_7b" if large_scale else "small_test_ckpt"

        cmd = f"""
        tune alpaca_generate
            --config {_CONFIG_PATH} \
            model=torchtune.models.{ckpt} \
            model_checkpoint={self._fetch_ckpt_model_path(ckpt)} \
            output_dir={tmpdir} \
        """.split()

        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")
