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
import torch
import torchtune
from tests.common import TUNE_PATH
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    get_loss_values_from_metric_logger,
    gpu_test,
)


CKPT = "llama2_7b"

# TODO: remove this once we have eval configs exposed properly
pkg_path = Path(torchtune.__file__).parent.absolute()
EVAL_CONFIG_PATH = Path.joinpath(
    pkg_path, "_cli", "eval_configs", "default_eval_config.yaml"
)


@gpu_test(gpu_count=4)
class TestFullFinetuneDistributed7BLoss:
    def _get_test_config_overrides(self):
        return [
            "batch_size=1",
            "tokenizer.path=/tmp/test-artifacts/tokenizer.model",
            "dataset.train_on_input=False",
            "seed=9",
            "epochs=2",
            "max_steps_per_epoch=2",
            "log_every_n_steps=1",
        ]

    def _fetch_expected_loss_values(self):
        return [1.1281, 1.8182, 1.2476, 0.9085]

    @pytest.mark.slow_integration_test
    def test_loss(self, tmpdir, monkeypatch):
        ckpt_path = Path(CKPT_MODEL_PATHS[CKPT])
        ckpt_dir = ckpt_path.parent
        cmd = f"""
        tune --nnodes 1 --nproc_per_node 2 full_finetune_distributed
            --config full_finetune_distributed \
            output_dir={tmpdir} \
            checkpointer=torchtune.utils.FullModelTorchTuneCheckpointer
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
        """.split()
        cmd = cmd + self._get_test_config_overrides()

        monkeypatch.setattr(sys, "argv", cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        loss_values = get_loss_values_from_metric_logger(tmpdir)
        expected_loss_values = self._fetch_expected_loss_values()
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-3, atol=1e-3
        )


@gpu_test(gpu_count=2)
class TestLoRA7BDistributedFinetuneEval:
    @pytest.mark.slow_integration_test
    def test_finetune_and_eval(self, tmpdir, capsys, monkeypatch):

        ckpt_path = Path(CKPT_MODEL_PATHS[CKPT])
        ckpt_dir = ckpt_path.parent

        # Run on prod LoRA FT config but with only 10 steps for now
        ft_cmd = f"""
        tune --nnodes 1 --nproc_per_node 2 lora_finetune_distributed
            --config lora_finetune_distributed \
            output_dir={tmpdir} \
            checkpointer=torchtune.utils.FullModelTorchTuneCheckpointer
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
            max_steps_per_epoch=500 \
        """.split()

        monkeypatch.setattr(sys, "argv", ft_cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        eval_cmd = f"""
        tune eval \
            --config {EVAL_CONFIG_PATH} \
            model_checkpoint={tmpdir}/torchtune_model_0.pt \
            tokenizer._component_=torchtune.models.llama2.llama2_tokenizer \
            tokenizer.path=/tmp/test-artifacts/tokenizer.model \
            tasks=['truthfulqa_mc2']
            limit=100 \
            device=cuda \
        """.split()
        monkeypatch.setattr(sys, "argv", eval_cmd)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        out_err = capsys.readouterr()
        acc = float(re.findall(r"'acc,none': (\d+\.\d+)", out_err.out)[0])
        assert acc >= 0.4
