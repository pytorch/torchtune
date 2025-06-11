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
    gpu_test,
    TOKENIZER_PATHS,
)

from torchtune.training.checkpointing._utils import (
    get_largest_iter_folder,
    RECIPE_STATE_DIRNAME,
    SHARD_FNAME,
)


class TestFullFinetuneSingleDeviceRecipe:
    def _get_test_config_overrides(self):
        return [
            "dtype=fp32",
            "enable_activation_checkpointing=False",
            "enable_activation_offloading=False",
            "dataset.train_on_input=False",
            "seed=9",
            "epochs=2",
            "max_steps_per_epoch=2",
            "optimizer=torch.optim.AdamW",
            "optimizer.lr=2e-5",
            "lr_scheduler.num_warmup_steps=0",
            "lr_scheduler.num_cycles=0",
            "log_every_n_steps=1",
        ] + dummy_alpaca_dataset_config()

    def _fetch_expected_loss_values(self, model_ckpt):
        loss_values_map = {"llama3_hf_138m": [11.8934, 11.9444, 11.8903, 11.8915]}

        return loss_values_map[model_ckpt]

    @pytest.mark.integration_test
    @pytest.mark.parametrize("compile", [True, False])
    @pytest.mark.parametrize(
        "micro_batch_size, gradient_accumulation_steps, optimizer_in_bwd",
        [(8, 1, True), (2, 4, False)],
    )
    @pytest.mark.parametrize(
        "config, model_ckpt",
        [
            ("llama3/8B_full_single_device", "llama3_hf_138m"),
        ],
    )
    @gpu_test(gpu_count=1)
    def test_loss(
        self,
        compile,
        micro_batch_size,
        gradient_accumulation_steps,
        optimizer_in_bwd,
        config,
        model_ckpt,
        tmpdir,
        monkeypatch,
    ):
        ckpt_dir = Path(CKPT_MODEL_PATHS[model_ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_ckpt])
        log_file = gen_log_file_name(tmpdir)

        cmd = f"""
        tune run full_finetune_single_device \
            --config {config} \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={log_file} \
            compile={compile} \
        """.split()

        model_config = MODEL_TEST_CONFIGS[model_ckpt]
        cmd = cmd + self._get_test_config_overrides() + model_config
        # "optimizer_in_bwd=True" would free gradient info before clip_grad, causing
        # wrong grad_norm, so we only test one of them each time. But loss values
        # should be the same.
        if not optimizer_in_bwd:
            cmd.append("clip_grad_norm=100")
            cmd.append("optimizer_in_bwd=False")
        else:
            cmd.append("optimizer_in_bwd=True")
        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Make sure to clear compile state in between tests
        if compile:
            torch._dynamo.reset()

        loss_values = get_loss_values_from_metric_logger(log_file)
        expected_loss_values = self._fetch_expected_loss_values(model_ckpt)

        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-3, atol=1e-3
        )

    @pytest.mark.integration_test
    @pytest.mark.parametrize(
        "optimizer_in_bwd",
        [True, False],
    )
    @gpu_test(gpu_count=1)
    @pytest.mark.parametrize(
        "model_ckpt",
        [
            ("llama3_hf_138m"),
        ],
    )
    def test_training_state_on_resume(
        self, tmpdir, monkeypatch, optimizer_in_bwd, model_ckpt
    ):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 2 epochs
            - Resume training after epoch 1
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        """

        ckpt_dir = Path(CKPT_MODEL_PATHS[model_ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_ckpt])
        first_log_file = gen_log_file_name(tmpdir, suffix="first")

        # Train for two epochs
        cmd_1 = f"""
        tune run full_finetune_single_device \
            --config llama3/8B_full_single_device \
            batch_size=8 \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={first_log_file} \
            optimizer_in_bwd={optimizer_in_bwd} \
        """.split()

        model_config = MODEL_TEST_CONFIGS[model_ckpt]
        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Sanity check that the loss values are expected for the initial run
        expected_loss_values = self._fetch_expected_loss_values(model_ckpt)
        loss_values = get_loss_values_from_metric_logger(first_log_file)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-3, atol=1e-3
        )

        # Resume training
        epoch_folder = get_largest_iter_folder(tmpdir)
        epoch_folder_minus_one = f"epoch_{int(epoch_folder.split('_')[-1]) - 1}"
        model_ckpt_fname = (
            SHARD_FNAME.format(cpt_idx="1".zfill(5), num_shards="1".zfill(5))
            + ".safetensors"
        )
        log_file = gen_log_file_name(tmpdir, suffix="resume")
        cmd_2 = f"""
        tune run full_finetune_single_device \
            --config llama3/8B_full_single_device \
            batch_size=8 \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{os.path.join(epoch_folder_minus_one, model_ckpt_fname)}]\
            checkpointer.recipe_checkpoint={os.path.join(RECIPE_STATE_DIRNAME, "recipe_state.pt")}\
            checkpointer.output_dir={tmpdir} \
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            resume_from_checkpoint=True \
            metric_logger.filename={log_file} \
            optimizer_in_bwd={optimizer_in_bwd} \
        """.split()

        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_loss_values = self._fetch_expected_loss_values(model_ckpt)[2:]

        loss_values = get_loss_values_from_metric_logger(log_file)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.integration_test
    @gpu_test(gpu_count=1)
    @pytest.mark.parametrize(
        "model_ckpt",
        [
            ("llama3_hf_138m"),
        ],
    )
    def test_training_state_on_resume_with_async_checkpointing(
        self, tmpdir, monkeypatch, model_ckpt
    ):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 2 epochs
            - Resume training after epoch 1
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        """

        ckpt_dir = Path(CKPT_MODEL_PATHS[model_ckpt])
        tokenizer_path = Path(TOKENIZER_PATHS[model_ckpt])
        first_log_file = gen_log_file_name(tmpdir, suffix="first")

        # Train for two epochs
        cmd_1 = f"""
        tune run full_finetune_single_device \
            --config llama3/8B_full_single_device \
            batch_size=8 \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            enable_async_checkpointing=True\
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            metric_logger.filename={first_log_file} \
            optimizer_in_bwd=False \
        """.split()

        model_config = MODEL_TEST_CONFIGS[model_ckpt]
        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Sanity check that the loss values are expected for the initial run
        expected_loss_values = self._fetch_expected_loss_values(model_ckpt)
        loss_values = get_loss_values_from_metric_logger(first_log_file)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )

        # Resume training
        log_file = gen_log_file_name(tmpdir, suffix="resume")
        cmd_2 = f"""
        tune run full_finetune_single_device \
            --config llama3/8B_full_single_device \
            batch_size=8 \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            enable_async_checkpointing=True\
            tokenizer.path='{tokenizer_path}' \
            tokenizer.prompt_template=null \
            resume_from_checkpoint=True \
            metric_logger.filename={log_file} \
            optimizer_in_bwd=False \
        """.split()

        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_loss_values = self._fetch_expected_loss_values(model_ckpt)[2:]

        loss_values = get_loss_values_from_metric_logger(log_file)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )
