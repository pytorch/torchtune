# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re

import runpy

import shutil

import sys
from pathlib import Path

import pytest
import torch
from tests.common import TUNE_PATH

from tests.recipes.utils import dummy_alpaca_dataset_config, MODEL_TEST_CONFIGS
from tests.test_utils import (
    CKPT_MODEL_PATHS,
    gen_log_file_name,
    get_loss_values_from_metric_logger,
    gpu_test,
    TOKENIZER_PATHS,
)

from torchtune.training.checkpointing._utils import get_largest_iter_folder


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
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.integration_test
    @pytest.mark.parametrize("keep_last_n_checkpoints", [1, 2])
    @pytest.mark.parametrize("save_every_n_steps", [1, 2])
    def test_checkpointing_with_steps(
        self, tmpdir, monkeypatch, keep_last_n_checkpoints, save_every_n_steps
    ):
        model_type = "llama3_hf_138m"
        ckpt_dir = Path(CKPT_MODEL_PATHS[model_type])
        tokenizer_path = Path(TOKENIZER_PATHS[model_type])
        model_config = MODEL_TEST_CONFIGS[model_type]

        # Train for two epochs (anywhere from 2 -> 4 ckpts)
        cmd_1 = f"""
        tune run full_finetune_single_device \
            --config llama3/8B_full_single_device  \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.keep_last_n_checkpoints={keep_last_n_checkpoints} \
            save_every_n_steps={save_every_n_steps} \
            tokenizer.path={tokenizer_path} \
            tokenizer.prompt_template=null \
        """.split()
        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        regex_to_match = re.compile("step_([0-9]+)")
        # Iterate over the directory contents, find all directories that match
        # `regex_to_match`. Assert that the number of directories found is equal
        # to the `keep_last_n_checkpoints` value. Also assert that each checkpoint
        # number is a multiple of `save_every_n_steps`.
        ckpt_dirs = [
            d
            for d in os.listdir(tmpdir)
            if os.path.isdir(os.path.join(tmpdir, d)) and regex_to_match.match(d)
        ]
        assert len(ckpt_dirs) == keep_last_n_checkpoints
        for ckpt_dir in ckpt_dirs:
            step = int(regex_to_match.match(ckpt_dir).group(1))
            assert step % save_every_n_steps == 0

        # Also make sure that the last checkpoint has the correct number of steps
        most_recent_checkpoint = get_largest_iter_folder(tmpdir, pattern=r"^step_(\d+)")
        step = int(regex_to_match.match(most_recent_checkpoint).group(1))
        assert step == 4  # 2 epochs * 2 steps per epoch

    # test does not work without using shutil in order to remove the last directory
    # (otherwise HF checkpointer looks for latest directory that does not have recipe_state.pt)
    # requires recipe_checkpoint to be specified even though that should be deprecated ?
    @pytest.mark.integration_test
    @pytest.mark.parametrize("use_steps", [True, False])
    @gpu_test(gpu_count=1)
    def test_training_state_on_resume(self, tmpdir, use_steps, monkeypatch):
        """We want to be sure that now we use steps, we can resume correctly from a checkpoint.
        Once we fully transition to steps, we can remove the test above."""
        # 0. Set up variables
        model_type = "llama3_hf_138m"
        ckpt_dir = Path(CKPT_MODEL_PATHS[model_type])
        model_config = MODEL_TEST_CONFIGS[model_type]
        tokenizer_path = TOKENIZER_PATHS[model_type]
        log_file = gen_log_file_name(tmpdir)

        # 1. Train for two epochs, keep 2 checkpoints
        cmd_1 = f"""
        tune run full_finetune_single_device \
            --config llama3/8B_full_single_device \
            batch_size=8 \
            output_dir={tmpdir} \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[model.safetensors]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.keep_last_n_checkpoints=2 \
            tokenizer.path={tokenizer_path} \
            tokenizer.prompt_template=null \
            optimizer_in_bwd=False \
        """.split()
        if use_steps:
            cmd_1.append("save_every_n_steps=2")
            final_ckpt_dir = "step_4"
            prev_ckpt_dir = "step_2"
        else:
            final_ckpt_dir = "epoch_1"
            prev_ckpt_dir = "epoch_0"
        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # 2. Find the checkpoint at the end of the first epoch
        suffix = ".safetensors"
        model_ckpt_fname = "model" + suffix
        assert os.path.exists(
            os.path.join(tmpdir, prev_ckpt_dir, model_ckpt_fname)
        ), "Checkpoint file does not exist"

        shutil.rmtree(tmpdir / final_ckpt_dir)

        # 3. Resume training w/ the checkpoint from epoch boundary
        cmd_2 = f"""
        tune run full_finetune_single_device \
            --config llama3/8B_full_single_device \
            batch_size=8 \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.training.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir={tmpdir}/{prev_ckpt_dir} \
            checkpointer.checkpoint_files=["{os.path.join(tmpdir, prev_ckpt_dir, model_ckpt_fname)}"]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.recipe_checkpoint="recipe_state.pt"
            tokenizer.path={tokenizer_path} \
            tokenizer.prompt_template=null \
            resume_from_checkpoint=True \
            metric_logger.filename={log_file} \
            optimizer_in_bwd=False \
            save_every_n_steps=2 \
        """.split()
        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config
        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit, match=""):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # 4. Make sure loss values match the expected values
        expected_loss_values = self._fetch_expected_loss_values(model_type)[2:]
        loss_values = get_loss_values_from_metric_logger(log_file)

        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )
