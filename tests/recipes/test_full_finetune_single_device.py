# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os

import runpy

import sys
from pathlib import Path

import numpy as np

import pytest

import torch
from tests.common import TUNE_PATH

from tests.recipes.utils import (
    fetch_ckpt_model_path,
    get_checkpointer_class_path_for_test_ckpt,
    get_loss_values_from_metric_logger,
    llama2_test_config,
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
            "lr_scheduler=torchtune.modules.get_cosine_schedule_with_warmup",
            "lr_scheduler.num_warmup_steps=100",
        ]

    def _fetch_expected_loss_values(self, ckpt):
        small_test_ckpt_loss_values = [10.5074, 10.5563, 10.5152, 10.4851]
        llama2_7b_ckpt_loss_values = [1.1333, 1.1199, 1.2614, 0.9486]
        return (
            llama2_7b_ckpt_loss_values if "7b" in ckpt else small_test_ckpt_loss_values
        )

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
        tune full_finetune_single_device
            --config alpaca_llama2_full_finetune_single_device \
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
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        loss_values = get_loss_values_from_metric_logger(tmpdir)
        torch.testing.assert_close(
            loss_values, expected_loss_values, rtol=1e-4, atol=1e-4
        )

    def test_training_state_on_resume(self, tmpdir, monkeypatch):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 2 epochs
            - Resume training after epoch 1
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        """

        model_ckpt = "small_test_ckpt_hf"

        ckpt_path = Path(fetch_ckpt_model_path(model_ckpt))
        ckpt_dir = ckpt_path.parent

        # config file needed for model conversion. Since this is a really small json
        # this can be written within the test instead of downloading from s3.
        # We need two copies one for initial read and one for resume
        config = {
            "hidden_size": 256,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        }
        config_file = Path.joinpath(Path(ckpt_dir), "config.json")
        with config_file.open("w") as f:
            json.dump(config, f)

        # Train for two epochs
        cmd_1 = f"""
        tune full_finetune_single_device
            --config alpaca_llama2_full_finetune_single_device \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.utils.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            epochs=2 \
            log_every_n_steps=1 \
        """.split()

        model_config = llama2_test_config()
        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # We don't care about these loss values, just remove the log file
        _ = get_loss_values_from_metric_logger(tmpdir, remove_found_file=True)

        config_file = Path.joinpath(Path(tmpdir), "config.json")
        with config_file.open("w") as f:
            json.dump(config, f)

        # Resume training
        cmd_2 = f"""
        tune full_finetune_single_device
            --config alpaca_llama2_full_finetune_single_device \
            output_dir={tmpdir} \
            checkpointer._component_=torchtune.utils.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir={tmpdir} \
            checkpointer.checkpoint_files=[{os.path.join(tmpdir, "hf_model_0001_0.pt")}]\
            checkpointer.recipe_checkpoint={os.path.join(tmpdir, "recipe_state.pt")}
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            resume_from_checkpoint=True \
        """.split()

        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_loss_values = self._fetch_expected_loss_values(model_ckpt)[2:]

        loss_values = get_loss_values_from_metric_logger(tmpdir)
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
            "dataset.train_on_input=False",
            "seed=9",
            "optimizer=torch.optim.AdamW",
            "optimizer.lr=2e-5",
            "lr_scheduler=torchtune.modules.get_cosine_schedule_with_warmup",
            "lr_scheduler.num_warmup_steps=100",
        ]

    @pytest.mark.parametrize("full_batch_size, micro_batch_size", [(2, 1), (4, 1)])
    def test_gradient_accumulation(
        self, full_batch_size, micro_batch_size, capsys, mocker, tmpdir, monkeypatch
    ):
        """Test whether gradient accumulation runs properly in the recipe. In general
        the sum of loss across minibatches should equal the loss over the full batch,
        but since our loss is normalized by the number of unmasked tokens, this does not
        hold in for our case. We use a dummy dataset where all tokens are unmasked, and
        in this test check that a single batch size of N yields the same loss as N accumulated
        microbatches of size 1.
        """
        model_ckpt = "small_test_ckpt_tune"
        gradient_accumulation_steps = full_batch_size // micro_batch_size

        ckpt_path = Path(fetch_ckpt_model_path(model_ckpt))
        ckpt_dir = ckpt_path.parent

        cmd_1 = f"""
        tune full_finetune_single_device \
            --config alpaca_llama2_full_finetune_single_device \
            checkpointer._component_=torchtune.utils.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir={ckpt_dir} \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            dataset=tests.recipes.utils.DummyDataset \
            batch_size={full_batch_size} \
            epochs=1 \
            max_steps_per_epoch=1 \
            output_dir={tmpdir} \
            log_every_n_steps=1 \
        """.split()

        model_config = llama2_test_config()
        cmd_1 = cmd_1 + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        no_accum_loss = get_loss_values_from_metric_logger(
            tmpdir, remove_found_file=True
        )[0]

        # Update the cmd with new values for gradient accumulation
        cmd_2 = f"""
        tune full_finetune_single_device \
            --config alpaca_llama2_full_finetune_single_device \
            checkpointer._component_=torchtune.utils.FullModelTorchTuneCheckpointer \
            checkpointer.checkpoint_dir={ckpt_dir} \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            dataset=tests.recipes.utils.DummyDataset \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            epochs=1 \
            max_steps_per_epoch=1 \
            output_dir={tmpdir} \
            log_every_n_steps=1 \
        """.split()
        cmd_2 = cmd_2 + self._get_test_config_overrides() + model_config

        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        accum_loss = np.mean(get_loss_values_from_metric_logger(tmpdir))
        torch.testing.assert_close(no_accum_loss, accum_loss, atol=1e-5, rtol=1e-5)
