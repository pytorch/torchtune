# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json
import logging
import os

import runpy

import sys
from pathlib import Path
from typing import Dict

import numpy as np

import pytest

import torch
from tests.common import TUNE_PATH

from tests.recipes.common import RECIPE_TESTS_DIR
from tests.recipes.utils import (
    fetch_ckpt_model_path,
    fetch_loss_values,
    llama2_small_test_ckpt,
    llama2_tiny_test_ckpt,
    validate_loss_values,
)
from tests.test_utils import get_assets_path, single_box_init

from torchtune import models

models.small_test_ckpt_tune = llama2_small_test_ckpt
models.small_test_ckpt_meta = llama2_small_test_ckpt
models.small_test_ckpt_hf = llama2_small_test_ckpt
models.llama2_tiny_test_ckpt = llama2_tiny_test_ckpt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CONFIG_PATH = RECIPE_TESTS_DIR / "full_finetune_test_config.yaml"

_ASSETS = get_assets_path()

# Generating `tiny_llama2_checkpoint.pt`
# >>> import torch
# >>> from torchtune.models.llama2 import llama2
# >>> from tests.test_utils import fixed_init_model
# >>> super_small_llama2 = llama2(
# ... vocab_size=100,
# ... num_layers=2,
# ... num_heads=4,
# ... embed_dim=64,
# ... max_seq_len=64,
# ... norm_eps=1e-5,
# ... num_kv_heads=2,
# ... )
# >>> fixed_init_model(super_small_llama2, max_val=10.0, nonlinear=True)
# >>> torch.save(super_small_llama2.state_dict()}, "tiny_llama2_checkpoint.pt")


class TestFullFinetuneRecipe:
    def _fetch_expected_loss_values(self, ckpt) -> Dict[str, float]:
        small_test_ckpt_loss_values = {
            "1|1|": 10.5074,
            "1|2|": 10.5563,
            "2|1|": 10.5152,
            "2|2|": 10.4851,
        }
        llama2_7b_ckpt_loss_values = {
            "1|1|": 1.1333,
            "1|2|": 1.1199,
            "2|1|": 1.2614,
            "2|2|": 0.9486,
        }
        if (
            ckpt == "small_test_ckpt_tune"
            or ckpt == "small_test_ckpt_meta"
            or ckpt == "small_test_ckpt_hf"
        ):
            return small_test_ckpt_loss_values
        if ckpt == "llama2.llama2_7b":
            return llama2_7b_ckpt_loss_values
        raise ValueError(f"Unknown ckpt {ckpt}")

    def fetch_checkpointer(self, ckpt):
        if ckpt == "small_test_ckpt_tune" or ckpt == "llama2.llama2_7b":
            return "FullModelTorchTuneCheckpointer"
        if ckpt == "small_test_ckpt_hf":
            return "FullModelHFCheckpointer"
        if ckpt == "small_test_ckpt_meta":
            return "FullModelMetaCheckpointer"

    @pytest.mark.parametrize("single_device", [True])
    def test_loss(self, single_device, capsys, pytestconfig, tmpdir, monkeypatch):
        large_scale = pytestconfig.getoption("--large-scale")
        ckpts = (
            ["llama2.llama2_7b"]
            if large_scale
            else [
                "small_test_ckpt_tune",
                "small_test_ckpt_hf",
                "small_test_ckpt_meta",
            ]
        )

        for ckpt in ckpts:
            expected_loss_values = self._fetch_expected_loss_values(ckpt)
            ckpt_path = Path(fetch_ckpt_model_path(ckpt))
            ckpt_dir = ckpt_path.parent
            checkpointer = self.fetch_checkpointer(ckpt)

            if ckpt == "small_test_ckpt_hf":
                config = {
                    "hidden_size": 256,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                }
                config_file = Path.joinpath(Path(ckpt_dir), "config.json")
                with config_file.open("w") as f:
                    json.dump(config, f)

            if single_device:
                recipe_cmd = "full_finetune_single_device"
            else:
                recipe_cmd = "full_finetune_distributed"
            cmd = f"""
            tune {recipe_cmd}
                --config {_CONFIG_PATH} \
                output_dir={tmpdir} \
                model=torchtune.models.{ckpt} \
                checkpointer._component_=torchtune.utils.{checkpointer}
                checkpointer.checkpoint_dir='{ckpt_dir}' \
                checkpointer.checkpoint_files=[{ckpt_path}]\
                checkpointer.output_dir={tmpdir} \
                checkpointer.model_type=LLAMA2 \
                log_every_n_steps=1
            """.split()

            monkeypatch.setattr(sys, "argv", cmd)
            with pytest.raises(SystemExit):
                with (
                    single_box_init(init_pg=False)
                    if not single_device
                    else contextlib.nullcontext()
                ):
                    runpy.run_path(TUNE_PATH, run_name="__main__")

            loss_values = fetch_loss_values(capsys.readouterr().err)
            validate_loss_values(loss_values, expected_loss_values)
            capsys.readouterr()

    def test_training_state_on_resume(self, capsys, tmpdir, monkeypatch):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 4 epochs
            - Resume training after epoch 3
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        """

        model_ckpt = "small_test_ckpt_hf"
        expected_loss_values = self._fetch_expected_loss_values(model_ckpt)

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

        # Train
        cmd_1 = f"""
        tune full_finetune_single_device
            --config {_CONFIG_PATH} \
            output_dir={tmpdir} \
            model=torchtune.models.{model_ckpt} \
            checkpointer._component_=torchtune.utils.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir='{ckpt_dir}' \
            checkpointer.checkpoint_files=[{ckpt_path}]\
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            epochs=4 \
            log_every_n_steps=1 \
        """.split()

        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Clear stdout
        capsys.readouterr()

        config_file = Path.joinpath(Path(tmpdir), "config.json")
        with config_file.open("w") as f:
            json.dump(config, f)

        # Resume training
        cmd_2 = f"""
        tune full_finetune_single_device
            --config {_CONFIG_PATH} \
            output_dir={tmpdir} \
            model=torchtune.models.{model_ckpt} \
            checkpointer._component_=torchtune.utils.FullModelHFCheckpointer \
            checkpointer.checkpoint_dir={tmpdir} \
            checkpointer.checkpoint_files=[{os.path.join(tmpdir, "hf_model_0001_2.pt")}]\
            checkpointer.recipe_checkpoint={os.path.join(tmpdir, "recipe_state.pt")}
            checkpointer.output_dir={tmpdir} \
            checkpointer.model_type=LLAMA2 \
            epochs=4 \
            resume_from_checkpoint=True \
            max_steps_per_epoch=None \
            seed=0 \
            log_every_n_steps=1 \
        """.split()

        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        expected_loss_values = {
            "4|1|": 10.4905,
            "4|2|": 10.5057,
        }

        loss_values = fetch_loss_values(capsys.readouterr().err)
        validate_loss_values(loss_values, expected_loss_values)


class TestRecipeGradientAccumulation:
    @pytest.mark.parametrize("full_batch_size, micro_batch_size", [(2, 1), (4, 1)])
    def test_gradient_accumulation(
        self, full_batch_size, micro_batch_size, capsys, mocker, tmpdir, monkeypatch
    ):
        # We use a tiny model to reduce the error accumulation in the test
        # It's impossible to make a large model produce the same loss values
        # in the same way as the full batch size.
        model_ckpt = "small_test_ckpt_tune"
        gradient_accumulation_steps = full_batch_size // micro_batch_size

        ckpt_path = Path(fetch_ckpt_model_path(model_ckpt))
        ckpt_dir = ckpt_path.parent

        cmd = f"""
        tune full_finetune_single_device \
            --config {_CONFIG_PATH} \
            model=torchtune.models.{model_ckpt} \
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

        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # the first run assumes the complete batch and so we have a single loss value
        loss_value = float(
            [
                value
                for key, value in fetch_loss_values(capsys.readouterr().err).items()
            ][0]
        )
        # Update the cmd with new values for gradient accumulation
        cmd_2 = f"""
        tune full_finetune_single_device \
            --config {_CONFIG_PATH} \
            model=torchtune.models.{model_ckpt} \
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

        monkeypatch.setattr(sys, "argv", cmd_2)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        acc_loss_value = np.mean(
            [
                float(value)
                for key, value in fetch_loss_values(capsys.readouterr().err).items()
            ]
        )
        torch.testing.assert_close(loss_value, acc_loss_value, atol=1e-5, rtol=1e-5)
