# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import runpy

import sys
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
from tests.test_utils import get_assets_path

from torchtune import models

models.small_test_ckpt = llama2_small_test_ckpt
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
# >>> torch.save({"model": super_small_llama2.state_dict()}, "tiny_llama2_checkpoint.pt")

_CONFIG_PATH = RECIPE_TESTS_DIR / "full_finetune_test_config.yaml"


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
        if ckpt == "small_test_ckpt":
            return small_test_ckpt_loss_values
        if ckpt == "llama2.llama2_7b":
            return llama2_7b_ckpt_loss_values
        raise ValueError(f"Unknown ckpt {ckpt}")

    def test_loss(self, capsys, pytestconfig, tmpdir, monkeypatch):
        large_scale = pytestconfig.getoption("--large-scale")
        ckpt = "llama2.llama2_7b" if large_scale else "small_test_ckpt"
        expected_loss_values = self._fetch_expected_loss_values(ckpt)

        cmd = f"""
        tune full_finetune
            --config {_CONFIG_PATH} \
            --override \
            output_dir={tmpdir} \
            model._component_=torchtune.models.{ckpt} \
            model_checkpoint={fetch_ckpt_model_path(ckpt)} \
        """.split()

        monkeypatch.setattr(sys, "argv", cmd)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        loss_values = fetch_loss_values(capsys.readouterr().err)
        validate_loss_values(loss_values, expected_loss_values)

    def test_training_state_on_resume(self, capsys, tmpdir, monkeypatch):
        """Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of three stages:
            - Train a model for 4 epochs
            - Resume training after epoch 3
            - Make sure final loss matches the expected value of a model successfully resumed from a ckpt
        """

        model_ckpt = "small_test_ckpt"
        expected_loss_values = self._fetch_expected_loss_values(model_ckpt)

        # Train
        cmd_1 = f"""
        tune full_finetune
            --config {_CONFIG_PATH} \
            --override \
            output_dir={tmpdir} \
            model._component_=torchtune.models.{model_ckpt} \
            model_checkpoint={fetch_ckpt_model_path(model_ckpt)} \
            epochs=4 \
        """.split()

        monkeypatch.setattr(sys, "argv", cmd_1)
        with pytest.raises(SystemExit):
            runpy.run_path(TUNE_PATH, run_name="__main__")

        # Clear stdout
        capsys.readouterr()

        # Resume training
        cmd_2 = f"""
        tune full_finetune
            --config {_CONFIG_PATH} \
            --override \
            output_dir={tmpdir} \
            model._component_=torchtune.models.{model_ckpt} \
            model_checkpoint={os.path.join(tmpdir, "model_2.ckpt")} \
            epochs=4 \
            resume_from_checkpoint=True \
            max_steps_per_epoch=None \
            seed=0 \
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


@pytest.fixture
def create_mock_setup_data_fn(mocker):
    mocker.patch(
        "recipes.full_finetune.FullFinetuneRecipe._setup_data",
        wraps=dummy_setup_data_fn,
    )


class TestRecipeGradientAccumulation:
    @pytest.mark.parametrize("full_batch_size, micro_batch_size", [(2, 1), (4, 1)])
    def test_gradient_accumulation(
        self, full_batch_size, micro_batch_size, capsys, mocker, tmpdir, monkeypatch
    ):
        # We use a tiny model to reduce the error accumulation in the test
        # It's impossible to make a large model produce the same loss values
        # in the same way as the full batch size.
        model_ckpt = "llama2_tiny_test_ckpt"
        gradient_accumulation_steps = full_batch_size // micro_batch_size

        cmd = f"""
        tune full_finetune \
            --config {_CONFIG_PATH} \
            --override \
            model._component_=torchtune.models.{model_ckpt} \
            model_checkpoint={fetch_ckpt_model_path(model_ckpt)} \
            dataset._component_=tests.recipes.utils.DummyDataset \
            batch_size={full_batch_size} \
            epochs=1 \
            max_steps_per_epoch=1 \
            output_dir={tmpdir} \
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
        tune full_finetune \
            --config {_CONFIG_PATH} \
            --override \
            model._component_=torchtune.models.{model_ckpt} \
            model_checkpoint={fetch_ckpt_model_path(model_ckpt)} \
            dataset._component_=tests.recipes.utils.DummyDataset \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            epochs=1 \
            max_steps_per_epoch=1 \
            output_dir={tmpdir} \
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
