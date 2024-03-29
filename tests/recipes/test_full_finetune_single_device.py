# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import runpy

import sys
from pathlib import Path

import numpy as np

import pytest

import torch
from tests.common import TUNE_PATH

from tests.recipes.utils import llama2_test_config, write_hf_ckpt_config
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    gen_log_file_name,
    get_loss_values_from_metric_logger,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFullFinetuneSingleDeviceRecipe:
    def _get_test_config_overrides(self):
        return [
            "batch_size=8",
            "device=cpu",
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

    def _fetch_expected_loss_values(self):
        return [10.5074, 10.5563, 10.5152, 10.4851]

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "config", ["full_single_device_low_memory", "full_single_device"]
    )
    def test_loss(self, config, tmpdir, monkeypatch):
        ckpt = "small_test_ckpt_meta"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        cmd = f"""
        tune full_finetune_single_device
            --config llama2/7B_{config} \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.utils.FullModelMetaCheckpointer
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            metric_logger.filename={log_file} \
        """.split()

        model_config = llama2_test_config()
        cmd = cmd + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        loss_values = get_loss_values_from_metric_logger(log_file)
        expected_loss_values = self._fetch_expected_loss_values()
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
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

        ckpt = "small_test_ckpt_hf"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        log_file = gen_log_file_name(tmpdir)

        # Config file needed for model conversion.
        # Create a second copy for training resume
        write_hf_ckpt_config(ckpt_dir)
        write_hf_ckpt_config(tmpdir)

        # Train for two epochs
        cmd_1 = f"""
        tune full_finetune_single_device
            --config llama2/7B_full_single_device \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.utils.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
        """.split()

        model_config = llama2_test_config()
        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Resume training
        cmd_2 = f"""
        tune full_finetune_single_device
            --config llama2/7B_full_single_device \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.utils.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir={tmpdir} \
            checkpointer.checkpoint_files=[{os.path.join(tmpdir, "hf_model_0001_0.pt")}]\
            checkpointer.recipe_checkpoint={os.path.join(tmpdir, "recipe_state.pt")}
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            resume_from_checkpoint=True \
            metric_logger.filename={log_file} \
        """.split()

        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_loss_values = self._fetch_expected_loss_values()[2:]

        loss_values = get_loss_values_from_metric_logger(log_file)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )


class TestFullFinetuneSingleDeviceGradientAccumulation:
    def _get_test_config_overrides(self):
        return [
            "device=cpu",
            "dtype=fp32",
            "enable_activation_checkpointing=False",
            "tokenizer.path=/tmp/test-artifacts/tokenizer.model",
            "dataset=tests.recipes.utils.DummyDataset",
            "dataset.train_on_input=False",
            "seed=9",
            "epochs=1",
            "max_steps_per_epoch=1",
            "optimizer=torch.optim.AdamW",
            "optimizer.lr=2e-5",
            "log_every_n_steps=1",
        ]

    @pytest.mark.integration_test
    def test_gradient_accumulation(self, tmpdir, monkeypatch):
        """Test whether gradient accumulation runs properly in the recipe. In general
        the sum of loss across minibatches should equal the loss over the full batch,
        but since our loss is normalized by the number of unmasked tokens, this does not
        hold in for our case. We use a dummy dataset where all tokens are unmasked, and
        in this test check that a single batch size of N yields the same loss as N accumulated
        microbatches of size 1.
        """
        full_batch_size = 4
        micro_batch_size = 1
        gradient_accumulation_steps = full_batch_size // micro_batch_size
        ckpt = "small_test_ckpt_tune"
        ckpt_path = Path(CKPT_MODEL_PATHS[ckpt])
        ckpt_dir = ckpt_path.parent
        no_grad_accum_log_file = gen_log_file_name(tmpdir, suffix="no_grad_accum")
        grad_accum_log_file = gen_log_file_name(tmpdir, suffix="grad_accum")

        cmd_1 = f"""
        tune full_finetune_single_device \
            --config llama2/7B_full_single_device \
            checkpointer._component_=torchtune.utils.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir={ckpt_dir} \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            batch_size={full_batch_size} \
            output_dir={tmpdir} \
            log_every_n_steps=1 \
            metric_logger.filename={no_grad_accum_log_file} \
        """.split()

        model_config = llama2_test_config()
        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        no_accum_loss = get_loss_values_from_metric_logger(no_grad_accum_log_file)[
            0
        ]  # List of a single element

        # Update the cmd with new values for gradient accumulation
        cmd_2 = f"""
        tune full_finetune_single_device \
            --config llama2/7B_full_single_device \
            checkpointer._component_=torchtune.utils.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir={ckpt_dir} \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            metric_logger.filename={grad_accum_log_file} \
        """.split()
        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        accum_loss = np.mean(get_loss_values_from_metric_logger(grad_accum_log_file))
        torch.testing.assert_close(no_accum_loss, accum_loss, atol=1e-5, rtol=1e-5)
