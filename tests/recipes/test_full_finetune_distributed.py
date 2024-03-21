# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import runpy

import sys
from pathlib import Path

import torch
from tests.common import TUNE_PATH

from tests.recipes.utils import (
    CKPT_MODEL_PATHS,
    get_loss_values_from_metric_logger,
    llama2_test_config,
    write_hf_ckpt_config,
)
from tests.test_utils import gpu_test

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFullFinetuneDistributedRecipe:
    def _get_test_config_overrides(self):
        return [
            "batch_size=4",
            "dtype=fp32",
            "enable_activation_checkpointing=False",
            "tokenizer.path=/tmp/test-artifacts/tokenizer.model",
            "dataset.train_on_input=False",
            "seed=9",
            "epochs=2",
            "max_steps_per_epoch=2",
            "optimizer=torch.optim.AdamW",
            "optimizer.lr=2e-5",
            "log_every_n_steps=1",
        ]

    def _fetch_expected_loss_values(self, ckpt):
        return [10.4574, 10.5872, 10.5092, 10.4756]

    @gpu_test(gpu_count=2)
    def test_loss(self, tmpdir, monkeypatch):
        ckpt = "small_test_ckpt_hf"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent

        # Config file needed for model conversion.
        write_hf_ckpt_config(ckpt_dir)

        cmd = f"""
        tune --nnodes 1 --nproc_per_node 2 full_finetune_distributed
            --config full_finetune_distributed \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.utils.FullModelHFCheckpointer
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
        """.split()
        model_config = llama2_test_config()
        cmd = cmd + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")
        loss_values = get_loss_values_from_metric_logger(tmpdir)
        expected_loss_values = self._fetch_expected_loss_values(ckpt)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )
