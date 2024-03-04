# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from copy import deepcopy
from typing import Dict

import pytest

import torch
from tests.recipes.utils import (
    fetch_ckpt_model_path,
    fetch_loss_values,
    llama2_small_test_ckpt,
    validate_loss_values,
)

from torch import nn

from torchtune import config, models, utils
from torchtune.datasets._alpaca import CROSS_ENTROPY_IGNORE_IDX

models.small_test_ckpt = llama2_small_test_ckpt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import runpy

import sys

from tests.common import TUNE_PATH

from tests.recipes.common import RECIPE_TESTS_DIR
from tests.test_utils import fixed_init_model

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


# Custom collate function reducing vocab size to build a smaller model
def custom_collate(
    batch,
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    reduced_vocab_dim: int = 10,
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

    def forward(self, x):
        embeddings = self.embed(x)
        out = self.out(embeddings)
        return out


def dummy_grad_accum_ckpt():
    with torch.device("cpu"):
        model = DummyModel()
        fixed_init_model(model)
    return model


def dummy_setup_data_fn(
    cfg_dataset,
    shuffle,
    batch_size,
):
    world_size, rank = utils.get_world_size_and_rank()
    ds = config.instantiate(
        cfg_dataset,
        tokenizer=self._tokenizer,
    )
    sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=0,
    )
    dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=partial(
            utils.padded_collate,
            padding_idx=0,  # Same padding_idx as custom collate function
            ignore_idx=CROSS_ENTROPY_IGNORE_IDX,
        ),
    )


models.dummy_grad_accum_ckpt = dummy_grad_accum_ckpt


@pytest.fixture
def create_mock_load_checkpoint(mocker):
    mocker.patch(
        "recipes.full_finetune.FullFinetuneRecipe.load_checkpoint",
        return_value={"model": None},
    )


@pytest.fixture
def create_mock_collate_fn(mocker):
    mocker.patch("torchtune.utils.padded_collate", wraps=custom_collate)


@pytest.fixture
def create_mock_setup_data_fn(mocker):
    mocker.patch(
        "recipes.full_finetune.FullFinetuneRecipe._setup_data",
        wraps=dummy_setup_data_fn,
    )


class TestRecipeGradientAccumulation:
    @pytest.mark.parametrize("full_batch_size, micro_batch_size", [(2, 1), (4, 1)])
    @pytest.mark.usefixtures("create_mock_load_checkpoint")
    @pytest.mark.usefixtures("create_mock_collate_fn")
    @pytest.mark.usefixtures("create_mock_setup_data_fn")
    def test_gradient_accumulation(
        self, full_batch_size, micro_batch_size, capsys, mocker, tmpdir, monkeypatch
    ):
        """
        Test gradient accumulation. Since this is model agnostic, we can just
        run this on a small dummy model.
        """

        model_ckpt = "dummy_grad_accum_ckpt"
        gradient_accumulation_steps = full_batch_size // micro_batch_size

        cmd = f"""
        tune full_finetune \
            --config {_CONFIG_PATH} \
            --override \
            model._component_=torchtune.models.{model_ckpt} \
            model_checkpoint=None \
            batch_size={full_batch_size} \
            epochs=1 \
            max_steps_per_epoch=1 \
            output_dir={tmpdir} \
        """.split()

        # Patch the recipe to use DummyModel class
        # Note that this cannot be done via a decorator because we use patch two separate times
        with mocker.patch(
            "recipes.full_finetune.FullFinetuneRecipe._setup_model",
            return_value=dummy_grad_accum_ckpt(),
        ):
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
            model_checkpoint=None \
            batch_size={micro_batch_size} \
            gradient_accumulation_steps={gradient_accumulation_steps} \
            epochs=1 \
            max_steps_per_epoch=1 \
            output_dir={tmpdir} \
        """.split()

        # Copy the dataloader and run a few iterations. CrossEntropyLoss is normalized
        # by the number of unmasked tokens, so we need to derive these values per sample
        # to appropriately compare losses with and without gradient accumulation.
        dummy_dataloader = deepcopy()
        normalization_factors = []
        for i, batch in enumerate(dummy_dataloader):
            labels = batch[1]
            num_unmasked_pos = (labels != CROSS_ENTROPY_IGNORE_IDX).sum().item()
            normalization_factors.append(num_unmasked_pos)
            if (i + 1) == full_batch_size:
                break

        # Patch the recipe to use DummyModel class. We use a separate patch
        # because otherwise the model params would remain the same from the baseline
        with mocker.patch(
            "recipes.full_finetune.FullFinetuneRecipe._setup_model",
            return_value=dummy_grad_accum_ckpt(),
        ):
            monkeypatch.setattr(sys, "argv", cmd_2)
            with pytest.raises(SystemExit):
                runpy.run_path(TUNE_PATH, run_name="__main__")

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
