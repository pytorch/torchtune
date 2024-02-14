# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import tempfile
from typing import Dict

import pytest

from recipes.full_finetune import FullFinetuneRecipe
from recipes.params import FullFinetuneParams
from recipes.tests.utils import (
    default_recipe_kwargs,
    fetch_ckpt_model_path,
    fetch_loss_values,
    llama2_small_test_ckpt,
    validate_loss_values,
)

from tests.test_utils import assert_expected
from torchtune import models

models.small_test_ckpt = llama2_small_test_ckpt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        if ckpt == "llama2_7b":
            return llama2_7b_ckpt_loss_values
        raise ValueError(f"Unknown ckpt {ckpt}")

    def test_loss(self, capsys, pytestconfig):
        large_scale = pytestconfig.getoption("--large-scale")
        ckpt = "llama2_7b" if large_scale else "small_test_ckpt"
        expected_loss_values = self._fetch_expected_loss_values(ckpt)

        kwargs_values = default_recipe_kwargs(ckpt)

        recipe_params = FullFinetuneParams(**kwargs_values)

        recipe = FullFinetuneRecipe(recipe_params)
        recipe.setup(params=recipe_params)
        recipe.train()

        loss_values = fetch_loss_values(capsys.readouterr().err)
        validate_loss_values(loss_values, expected_loss_values)

    def test_training_state_on_resume(self):
        """
        Test whether the recipe state is correctly updated on resume. Since this
        is model agnostic, we should run this on the small model only. The test
        consists of two stages:
            - Train a model for 4 epochs
            - Resume training after epoch 3 and check training state.
        """

        model_ckpt = "small_test_ckpt"
        expected_loss_values = self._fetch_expected_loss_values(model_ckpt)

        with tempfile.TemporaryDirectory() as tmpdirname:

            kwargs_values = {
                "dataset": "AlpacaDataset",
                "seed": 9,
                "shuffle": True,
                "model": model_ckpt,
                "model_checkpoint": fetch_ckpt_model_path(model_ckpt),
                "tokenizer": "llama2_tokenizer",
                "tokenizer_checkpoint": "/tmp/test-artifacts/tokenizer.model",
                "epochs": 4,
                "max_steps_per_epoch": 2,
                "output_dir": tmpdirname,
                "device": "cpu",
                "resume_from_checkpoint": False,
                "enable_fsdp": False,
            }

            recipe_params = FullFinetuneParams(**kwargs_values)

            recipe = FullFinetuneRecipe(recipe_params)
            recipe.setup(params=recipe_params)
            recipe.train()
            recipe.cleanup()

            # In the new run, remove seed and max_steps_per_epoch and
            # check if these are correctly inferred from the checkpoint
            # Note this will raise some warnings in the logs, but is a
            # stronger test
            kwargs_values_resume = {
                "dataset": "AlpacaDataset",
                "shuffle": True,
                "model": model_ckpt,
                "model_checkpoint": os.path.join(tmpdirname, "model_2.ckpt"),
                "tokenizer": "llama2_tokenizer",
                "tokenizer_checkpoint": "/tmp/test-artifacts/tokenizer.model",
                "epochs": 4,
                "output_dir": tmpdirname,
                "device": "cpu",
                "resume_from_checkpoint": True,  # set to True to resume
                "enable_fsdp": False,
            }

            recipe_params = FullFinetuneParams(**kwargs_values_resume)

            recipe = FullFinetuneRecipe(recipe_params)
            recipe.setup(params=recipe_params)

            assert recipe.epochs_run == 3
            assert recipe.seed == kwargs_values["seed"]
            assert recipe.max_steps_per_epoch == kwargs_values["max_steps_per_epoch"]
            assert recipe.total_epochs == kwargs_values["epochs"]
            assert recipe.total_training_steps == (
                3 * kwargs_values["max_steps_per_epoch"]
            )

    @pytest.mark.parametrize("gradient_accumulation_steps", [(1, 4), (1, 8)])
    @pytest.mark.parametrize("batch_size", [(4, 1), (8, 1)])
    def test_gradient_accumulation(
        self, gradient_accumulation_steps, batch_size, capsys, pytestconfig
    ):
        """
        Test gradient accumulation. Since this is model agnostic, we should just
        run this on the small model.
        """
        model_ckpt = "small_test_ckpt"

        for i in range(len(gradient_accumulation_steps)):
            kwargs_values = {
                "dataset": "AlpacaDataset",
                "train_on_input": False,
                "seed": 9,
                "shuffle": True,
                "model": model_ckpt,
                "model_checkpoint": fetch_ckpt_model_path(model_ckpt),
                "tokenizer": "llama2_tokenizer",
                "tokenizer_checkpoint": "/tmp/test-artifacts/tokenizer.model",
                "batch_size": batch_size[0],  # parametrized in the test
                "lr": 2e-5,
                "epochs": 1,  # make sure to run for 1 epoch
                "max_steps_per_epoch": 1,
                "optimizer": "AdamW",
                "loss": "CrossEntropyLoss",
                "output_dir": "/tmp",
                "device": "cpu",
                "dtype": "fp32",
                "resume_from_checkpoint": False,
                "enable_fsdp": False,
                "enable_activation_checkpointing": False,
                "metric_logger_type": "DiskLogger",
                "gradient_accumulation_steps": gradient_accumulation_steps[
                    0
                ],  # parametrized in the test
            }

            recipe_params = FullFinetuneParams(**kwargs_values)

            grad_accum_recipe = FullFinetuneRecipe(recipe_params)
            grad_accum_recipe.setup(params=recipe_params)
            grad_accum_recipe.train()

            # the first run assumes the complete batch and so we have a single loss value
            loss_value = float(
                [
                    value
                    for key, value in fetch_loss_values(capsys.readouterr().err).items()
                ][0]
            )

            # Update the dict with new values
            kwargs_values["batch_size"] = batch_size[1]
            kwargs_values["gradient_accumulation_steps"] = gradient_accumulation_steps[
                1
            ]

            recipe_params = FullFinetuneParams(**kwargs_values)

            grad_accum_recipe = FullFinetuneRecipe(recipe_params)
            grad_accum_recipe.setup(params=recipe_params)
            grad_accum_recipe.train()

            # the second run accumulates losses and so sum these up to compare
            acc_loss_value = sum(
                [
                    float(value) / (gradient_accumulation_steps[1])
                    for key, value in fetch_loss_values(capsys.readouterr().err).items()
                ]
            )
            assert_expected(loss_value, acc_loss_value, rtol=1e-1)
