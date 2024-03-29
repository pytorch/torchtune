# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import sys

import pytest

from tests.common import TUNE_PATH
from tests.recipes.utils import llama2_test_config
from tests.test_utils import CKPT_MODEL_PATHS


class TestAlpacaGenerateRecipe:
    @pytest.mark.integration_test
    def test_alpaca_generate(self, tmpdir, monkeypatch):
        ckpt = "small_test_ckpt_tune"
        model_checkpoint = CKPT_MODEL_PATHS[ckpt]
        cmd = f"""
        tune run alpaca_generate \
            --config alpaca_generate \
            model_checkpoint={model_checkpoint} \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
            output_dir={tmpdir} \
        """.split()

        model_config = llama2_test_config()
        cmd += model_config

        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")
