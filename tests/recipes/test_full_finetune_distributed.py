# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy

import sys
from pathlib import Path

import pytest
import torch
from tests.common import TUNE_PATH

from tests.recipes.utils import (
    CKPT_COMPONENT_MAP,
    dummy_alpaca_dataset_config,
    MODEL_TEST_CONFIGS,
    write_hf_ckpt_config,
)
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    gen_log_file_name,
    get_loss_values_from_metric_logger,
    gpu_test,
    TOKENIZER_PATHS,
)


class TestFullFinetuneDistributedRecipe:
    def _get_test_config_overrides(self):
        return [
            "batch_size=4",
            "dtype=fp32",
            "enable_activation_checkpointing=False",
            "dataset.train_on_input=False",
            "seed=9",
            "epochs=2",
            "max_steps_per_epoch=2",
            "optimizer=torch.optim.AdamW",
            "optimizer.lr=2e-5",
            "log_every_n_steps=1",
        ] + dummy_alpaca_dataset_config()

    def _fetch_expected_loss_values(self, model_type):
        loss_values_map = {
            "llama2": [10.5136, 10.4813, 10.5088, 10.5250],
            "llama3": [12.0673, 11.9072, 11.9302, 11.9355],
        }
        return loss_values_map[model_type]

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config, model_type, ckpt_type, fsdp_sharding_strategy",
        [
            ("llama2/7B_full", "llama2", "hf", None),
            ("llama3/8B_full", "llama3", "tune", None),
            ("llama3/8B_full", "llama3", "tune", "NO_SHARD"),
        ],
    )
    @pytest.mark.parametrize("optim_in_bwd", [True, False])
    @gpu_test(gpu_count=2)
    def test_loss(
        self,
        config,
        model_type,
        ckpt_type,
        fsdp_sharding_strategy,
        optim_in_bwd,
        tmpdir,
        monkeypatch,
    ):
        ckpt_component = CKPT_COMPONENT_MAP[ckpt_type]
        ckpt = model_type + "_" + ckpt_type
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_type])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        write_hf_ckpt_config(ckpt_dir)

        cmd = f"""
        tune run --nnodes 1 --nproc_per_node 2 full_finetune_distributed \
            --config {config} \
            output_dir={tmpdir} \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={log_file} \
        """.split()
        if fsdp_sharding_strategy:
            cmd.append(f"fsdp_sharding_strategy={fsdp_sharding_strategy}")
        model_config = MODEL_TEST_CONFIGS[model_type]
        cmd = cmd + self._get_test_config_overrides() + model_config
        # "optimizer_in_bwd=True" would free gradient info before clip_grad, causing
        # wrong grad_norm, so we only test one of them each time. But loss values
        # should be the same.
        if not optim_in_bwd:
            cmd.append("clip_grad_norm=100")
        else:
            cmd.append("optimizer_in_bwd=True")

        monkeypatch.setattr(sys, "argv", cmd)
        runpy.run_path(TUNE_PATH, run_name="__main__")
        loss_values = get_loss_values_from_metric_logger(log_file)
        expected_loss_values = self._fetch_expected_loss_values(model_type)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )
