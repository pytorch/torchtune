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
    TOKENIZER_PATHS,
)


class TestLoRAFinetuneSingleDeviceRecipe:
    def _get_test_config_overrides(self, dtype_str: str = "fp32", epochs: int = 2):
        return [
            "batch_size=8",
            "device=cpu",
            f"dtype={dtype_str}",
            "enable_activation_checkpointing=False",
            "dataset.train_on_input=False",
            "seed=9",
            f"epochs={epochs}",
            "max_steps_per_epoch=2",
            "optimizer.lr=2e-5",
            "log_every_n_steps=1",
            "gradient_accumulation_steps=1",
            "clip_grad_norm=100",
        ] + dummy_alpaca_dataset_config()

    def _fetch_expected_loss_values(self, model_type):
        loss_values_map = {
            "llama3": [11.0651, 11.0577, 11.0540, 11.7671],
        }
        return loss_values_map[model_type]

    @pytest.mark.integration_test
    @pytest.mark.parametrize("compile", [True, False])
    @pytest.mark.parametrize(
        "config, model_type, ckpt_type",
        [
            ("llama3_1/kd_single_device", "llama3", "tune"),
        ],
    )
    def test_loss(self, compile, config, model_type, ckpt_type, tmpdir, monkeypatch):
        ckpt_component = CKPT_COMPONENT_MAP[ckpt_type]
        ckpt = model_type + "_" + ckpt_type
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_type])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        cmd = f"""
        tune run kd_single_device \
            --config {config} \
            output_dir={tmpdir} \
            checkpointer._component_={ckpt_component} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}] \
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type={model_type.upper()} \
            teacher_checkpointer._component_={ckpt_component} \
            teacher_checkpointer.checkpoint_dir='{ckpt_dir}' \
            teacher_checkpointer.checkpoint_files=[{ckpt_path}] \
            teacher_checkpointer.output_dir={tmpdir} \
            teacher_checkpointer.model_type={model_type.upper()} \
            ~model.intermediate_dim \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger._component_=torchtune.training.metric_logging.DiskLogger \
            metric_logger.filename={log_file} \
            compile={compile} \
            kd_loss._component_=torchtune.modules.loss.ForwardKLWithChunkedOutputLoss \
            kd_ratio=0.5 \
        """.split()

        model_config = MODEL_TEST_CONFIGS[model_type + "_lora"]
        teacher_config = [
            "teacher_" + config for config in MODEL_TEST_CONFIGS[model_type]
        ]

        cmd = (
            cmd
            + self._get_test_config_overrides(dtype_str="fp32")
            + model_config
            + teacher_config
        )
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Make sure to clear compile state in between tests
        if compile:
            torch._dynamo.reset()

        loss_values = get_loss_values_from_metric_logger(log_file)
        # only take the first loss
        num_losses = int(len(loss_values) / 4)  # 2 steps per epoch, 2 epochs
        loss_values = loss_values[0::num_losses]
        expected_loss_values = self._fetch_expected_loss_values(model_type)
        print(loss_values)
        print(expected_loss_values)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.integration_test
    def test_training_state_on_resume(self, tmpdir, monkeypatch):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 2 epochs
            - Resume training after epoch 1
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        """

        ckpt = "llama3_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)
        tokenizer_path = Path(TOKENIZER_PATHS["llama3"])

        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune run kd_single_device \
            --config llama3_1/kd_single_device \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            teacher_checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
            teacher_checkpointer.checkpoint_dir='{ckpt_dir}' \
            teacher_checkpointer.checkpoint_files=[{ckpt_path}] \
            teacher_checkpointer.output_dir={tmpdir} \
            teacher_checkpointer.model_type=LLAMA3 \
            ~model.intermediate_dim \
            tokenizer.path={tokenizer_path} \
            tokenizer.prompt_template=null \
            metric_logger._component_=torchtune.training.metric_logging.DiskLogger \
            kd_loss._component_=torchtune.modules.loss.ForwardKLWithChunkedOutputLoss \
            kd_ratio=0.5 \
        """.split()

        model_config = MODEL_TEST_CONFIGS["llama3_lora"]
        teacher_config = [
            "teacher_" + config for config in MODEL_TEST_CONFIGS["llama3"]
        ]

        cmd_1 = (
            cmd_1 + self._get_test_config_overrides() + model_config + teacher_config
        )
        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Resume training
        cmd_2 = f"""
        tune run kd_single_device \
            --config llama3_1/kd_single_device \
            output_dir={tmpdir} \
            checkpointer=torchtune.training.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir={tmpdir} \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.adapter_checkpoint={os.path.join(tmpdir, "adapter_0.pt")}
            checkpointer.recipe_checkpoint={os.path.join(tmpdir, "recipe_state.pt")}
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA3 \
            teacher_checkpointer._component_=torchtune.training.FullModelTorchTuneCheckpointer \
            teacher_checkpointer.checkpoint_dir='{ckpt_dir}' \
            teacher_checkpointer.checkpoint_files=[{ckpt_path}] \
            teacher_checkpointer.output_dir={tmpdir} \
            teacher_checkpointer.model_type=LLAMA3 \
            ~model.intermediate_dim \
            resume_from_checkpoint=True \
            metric_logger._component_=torchtune.training.metric_logging.DiskLogger \
            metric_logger.filename={log_file} \
            tokenizer.path={tokenizer_path} \
            tokenizer.prompt_template=null \
            kd_loss._component_=torchtune.modules.loss.ForwardKLWithChunkedOutputLoss \
            kd_ratio=0.5 \
        """.split()
        cmd_2 = (
            cmd_2
            + self._get_test_config_overrides(epochs=3)
            + model_config
            + teacher_config
        )
        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Second epoch only
        expected_loss_values = self._fetch_expected_loss_values("llama3")[2:]
        loss_values = get_loss_values_from_metric_logger(log_file)
        # only take the first loss
        num_losses = int(len(loss_values) / 4)  # 2 steps per epoch, 2 epochs
        loss_values = loss_values[0::num_losses][:2]

        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-5, atol=1e-5
        )
