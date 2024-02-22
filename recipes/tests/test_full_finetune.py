# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import tempfile
from copy import deepcopy
from typing import Dict
from unittest.mock import patch

import pytest

import torch
from recipes.full_finetune import FullFinetuneRecipe
from recipes.params.full_finetune import FullFinetuneParams
from recipes.tests.utils import (
    default_recipe_kwargs,
    fetch_ckpt_model_path,
    fetch_loss_values,
    llama2_small_test_ckpt,
    validate_loss_values,
)

from torch import nn
from torchtune import models
from torchtune.utils.collate import padded_collate

models.ALL_MODELS["small_test_ckpt"] = llama2_small_test_ckpt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from tests.test_utils import fixed_init_model


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
                "dataset": "alpaca",
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
                "dataset": "alpaca",
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


# Custom collate function reducing vocab size to build a smaller model
def custom_collate(
    batch, padding_idx: int = 0, ignore_idx: int = -100, reduced_vocab_dim: int = 10
):
    input_ids, labels = padded_collate(batch, padding_idx, ignore_idx)
    input_ids = torch.remainder(input_ids, reduced_vocab_dim)
    labels = torch.where(
        labels == ignore_idx, labels, torch.remainder(labels, reduced_vocab_dim)
    )
    return input_ids, labels


# Dummy model class containing just an nn.Embedding and nn.Linear
class DummyModel(nn.Module):
    def __init__(self, reduced_vocab_size=10, embed_dim=16):
        super().__init__()
        self.reduced_vocab_size = reduced_vocab_size
        self.embed = nn.Embedding(reduced_vocab_size, embed_dim)
        self.out = nn.Linear(embed_dim, reduced_vocab_size, bias=False)

    # @torch.no_grad()
    # def reduce_vocab_dim(self, x):
    #     x = torch.remainder(x, self.reduced_vocab_size)
    #     return x

    def forward(self, x):
        # x = self.reduce_vocab_dim(x)
        embeddings = self.embed(x)
        out = self.out(embeddings)
        return out


def dummy_grad_accum_ckpt(vocab_size: int = 32_000, embed_dim=16):
    model = DummyModel()
    fixed_init_model(model)
    return model


models.ALL_MODELS["dummy_grad_accum_ckpt"] = dummy_grad_accum_ckpt


@pytest.fixture
def create_mock_load_checkpoint(mocker):
    mocker.patch(
        "recipes.full_finetune.FullFinetuneRecipe.load_checkpoint",
        return_value={"model": None},
    )


@pytest.fixture
def create_mock_collate_fn(mocker):
    mocker.patch("torchtune.utils.padded_collate", wraps=custom_collate)


class TestRecipeGradientAccumulation:
    @pytest.mark.parametrize("full_batch_size, micro_batch_size", [(2, 1), (4, 1)])
    @pytest.mark.usefixtures("create_mock_load_checkpoint")
    @pytest.mark.usefixtures("create_mock_collate_fn")
    def test_gradient_accumulation(self, full_batch_size, micro_batch_size, capsys):
        """
        Test gradient accumulation. Since this is model agnostic, we can just
        run this on a small dummy model.
        """

        model_ckpt = "dummy_grad_accum_ckpt"
        gradient_accumulation_steps = full_batch_size // micro_batch_size
        kwargs_values = {
            "dataset": "alpaca",
            "train_on_input": False,
            "seed": 9,
            "shuffle": True,
            "model": model_ckpt,
            "model_checkpoint": None,
            "tokenizer": "llama2_tokenizer",
            "tokenizer_checkpoint": "/tmp/test-artifacts/tokenizer.model",
            "batch_size": full_batch_size,
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
            "metric_logger_type": "disk",
            "gradient_accumulation_steps": 1,
        }

        # First run without gradient accumulation
        baseline_params = kwargs_values.copy()
        baseline_recipe_params = FullFinetuneParams(**baseline_params)
        baseline_recipe = FullFinetuneRecipe(baseline_recipe_params)

        # Patch the recipe to use DummyModel class
        # Note that this cannot be done via a decorator because we use patch two separate times
        with patch(
            "recipes.full_finetune.FullFinetuneRecipe._setup_model",
            return_value=models.get_model("dummy_grad_accum_ckpt", device="cpu"),
        ):
            baseline_recipe.setup(params=baseline_recipe_params)
        baseline_recipe.train()

        # the first run assumes the complete batch and so we have a single loss value
        loss_value = float(
            [
                value
                for key, value in fetch_loss_values(capsys.readouterr().err).items()
            ][0]
        )

        # Update the dict with new values for gradient accumulation
        grad_accum_params = kwargs_values.copy()
        grad_accum_params["batch_size"] = micro_batch_size
        grad_accum_params["gradient_accumulation_steps"] = gradient_accumulation_steps
        grad_accum_recipe_params = FullFinetuneParams(**grad_accum_params)
        grad_accum_recipe = FullFinetuneRecipe(grad_accum_recipe_params)

        # Patch the recipe to use DummyModel class. We use a separate patch
        # because otherwise the model params would remain the same from the baseline
        with patch(
            "recipes.full_finetune.FullFinetuneRecipe._setup_model",
            return_value=models.get_model("dummy_grad_accum_ckpt", device="cpu"),
        ):
            grad_accum_recipe.setup(params=grad_accum_recipe_params)

        # Copy the dataloader and run a few iterations. CrossEntropyLoss is normalized
        # by the number of unmasked tokens, so we need to derive these values per sample
        # to appropriately compare losses with and without gradient accumulation.
        dummy_dataloader = deepcopy(grad_accum_recipe._dataloader)
        normalization_factors = []
        for i, batch in enumerate(dummy_dataloader):
            labels = batch[1]
            num_unmasked_pos = (labels != -100).sum().item()
            normalization_factors.append(num_unmasked_pos)
            if (i + 1) == full_batch_size:
                break

        grad_accum_recipe.train()

        # the second run accumulates losses and so sum these up to compare
        acc_loss_value = sum(
            [
                normalization_factors[i] * float(value)
                for i, value in enumerate(
                    fetch_loss_values(capsys.readouterr().err).values()
                )
            ]
        ) / sum(normalization_factors)

        torch.testing.assert_close(loss_value, acc_loss_value)
