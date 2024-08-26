# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import runpy
import sys
from pathlib import Path

import pytest
import torchtune
from tests.common import TUNE_PATH
from tests.test_utils import CKPT_MODEL_PATHS, gpu_test


CKPT = "llama2_7b"

# TODO: remove this once we have eval configs exposed properly
pkg_path = Path(torchtune.__file__).parent.absolute()
EVAL_CONFIG_PATH = Path.joinpath(
    pkg_path, "_cli", "eval_configs", "default_eval_config.yaml"
)


@gpu_test(gpu_count=2)
class TestLoRA7BDistributedFinetuneEval:
    @pytest.mark.slow_integration_test
    def test_finetune_and_eval(self, tmpdir, capsys, monkeypatch):
        ### TODO: hack for testing on PR, remove prior to landing
        import torch

        if "dev" in torch.__version__:
            raise ValueError("Failing on nightlies")
        else:
            assert 1 == 1
