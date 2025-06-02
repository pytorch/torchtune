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


class TestQATSingleDeviceRecipe:
    def _get_test_config_overrides(self):
        return [
            "dtype=bf16",
            "enable_activation_checkpointing=True",
            "enable_activation_offloading=False",
            "seed=9",
            "epochs=2",
            "max_steps_per_epoch=2",
            "optimizer=torch.optim.AdamW",
            "optimizer.lr=2e-5",
            "log_every_n_steps=1",
        ] + dummy_alpaca_dataset_config()

    def _fetch_expected_loss_values(self, model_type, ckpt_type):
        # logic here may need to be adjusted in the future
        if model_type == "llama2" and ckpt_type == "hf":
            return [
                10.596881866455078,
                10.715113639831543,
                10.57275104522705,
                10.497347831726074,
            ]

    @pytest.mark.integration_test
    @gpu_test(gpu_count=1)
    @pytest.mark.parametrize(
        "config, model_type, ckpt_type, micro_batch_size, gradient_accumulation_steps",
        [
            ("llama2/1B_qat_single_device", "llama2", "hf", 1, 1),
        ],
    )
    def test_loss(
        self,
        config,
        model_type,
        ckpt_type,
        micro_batch_size,
        gradient_accumulation_steps,
        tmpdir,
        monkeypatch,
    ):
        ckpt_component = CKPT_COMPONENT_MAP.get(
            ckpt_type, "torchtune.training.FullModelHFCheckpointer"
        )
        ckpt = model_type + "_" + ckpt_type
        ckpt_path = Path(CKPT_MODEL_PATHS.get(ckpt))
        tokenizer_path = Path(TOKENIZER_PATHS.get(model_type))
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        write_hf_ckpt_config(ckpt_dir)

        cmd = f"""
        tune run qat_single_device \
            --config {config} \
            output_dir={tmpdir} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}] \
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={log_file} \
        """.split()
        model_config = MODEL_TEST_CONFIGS.get(model_type)
        cmd = cmd + self._get_test_config_overrides() + model_config

        # specify intermediate_dim for test small llama2
        if model_type == "llama2":
            cmd.append("model.intermediate_dim=768")

        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        loss_values = get_loss_values_from_metric_logger(log_file)
        expected_losses = self._fetch_expected_loss_values(model_type, ckpt_type)
        torch.testing.assert_close(loss_values, expected_losses, rtol=1e-3, atol=1e-3)
