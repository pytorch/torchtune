# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging

import runpy

import sys
from pathlib import Path

import pytest

import torch
from tests.common import TUNE_PATH

from tests.recipes.utils import (
    fetch_ckpt_model_path,
    get_checkpointer_class_path_for_test_ckpt,
    get_loss_values_from_metric_logger,
    llama2_test_config,
)
from tests.test_utils import gpu_test

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFullFinetuneRecipe:
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
            "lr_scheduler=torchtune.modules.get_cosine_schedule_with_warmup",
            "lr_scheduler.num_warmup_steps=100",
        ]

    def _fetch_expected_loss_values(self, ckpt):
        small_test_ckpt_loss_values = [10.4574, 10.5872, 10.5092, 10.4756]
        llama2_7b_ckpt_loss_values = [1.2012, 1.0482, 1.3395, 0.9876]
        return (
            llama2_7b_ckpt_loss_values if "7b" in ckpt else small_test_ckpt_loss_values
        )

    @gpu_test(gpu_count=2)
    @pytest.mark.parametrize(
        "ckpt",
        [
            "small_test_ckpt_tune",
            "small_test_ckpt_hf",
            "small_test_ckpt_meta",
            "llama2.llama2_7b",
        ],
    )
    def test_loss(self, ckpt, capsys, pytestconfig, tmpdir, monkeypatch):
        large_scale = pytestconfig.getoption("--large-scale")
        if ckpt == "llama2.llama2_7b" and not large_scale:
            pytest.skip("Skipping large-scale test")

        expected_loss_values = self._fetch_expected_loss_values(ckpt)
        ckpt_path = Path(fetch_ckpt_model_path(ckpt))
        ckpt_dir = ckpt_path.parent
        checkpointer = get_checkpointer_class_path_for_test_ckpt(ckpt)

        if ckpt == "small_test_ckpt_hf":
            config = {
                "hidden_size": 256,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
            }
            config_file = Path.joinpath(Path(ckpt_dir), "config.json")
            with config_file.open("w") as f:
                json.dump(config, f)

        cmd = f"""
        tune --nnodes 1 --nproc_per_node 2 full_finetune_distributed
            --config alpaca_llama2_full_finetune_distributed \
            output_dir={tmpdir} \
            checkpointer._component_={checkpointer}
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            log_every_n_steps=1
        """.split()

        model_config = (
            llama2_test_config()
            if ckpt != "llama2.llama2_7b"
            else ["model=torchtune.models.llama2.llama2_7b"]
        )
        cmd = cmd + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")

        loss_values = get_loss_values_from_metric_logger(tmpdir)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )
