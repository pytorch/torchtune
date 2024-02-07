# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import tempfile
from typing import Dict, Optional

import pytest
import recipes.finetune_llm as finetune_llm
from recipes.full_finetune import FullFinetuneRecipe
from recipes.params import FullFinetuneParams
from torchtune import models
from torchtune.models.llama2 import llama2

from torchtune.modules import TransformerDecoder


def small_test_ckpt(max_batch_size: Optional[int] = None) -> TransformerDecoder:
    return llama2(
        vocab_size=32_000,
        num_layers=4,
        num_heads=16,
        embed_dim=256,
        max_seq_len=2048,
        norm_eps=1e-5,
        num_kv_heads=8,
        max_batch_size=max_batch_size,
    )


models.small_test_ckpt = small_test_ckpt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFinetuneLLMRecipe:
    def _fetch_loss_values(self, output) -> Dict[str, float]:
        lines = output.splitlines()
        loss_values = {}
        for line in lines:
            if "Loss:" in line:
                splits = line.split("Loss:")
                loss_value = float(splits[1].split(":")[0])
                loss_values[splits[0]] = loss_value
        return loss_values

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

    def _fetch_ckpt_model_path(self, ckpt) -> str:
        if ckpt == "small_test_ckpt":
            return "/tmp/test-artifacts/small-ckpt-01242024"
        if ckpt == "llama2_7b":
            return "/tmp/test-artifacts/llama2-7b-01242024"
        raise ValueError(f"Unknown ckpt {ckpt}")

    def test_finetune_llm_loss(self, capsys, pytestconfig):
        large_scale = pytestconfig.getoption("--large-scale")
        ckpt = "llama2_7b" if large_scale else "small_test_ckpt"
        expected_loss_values = self._fetch_expected_loss_values(ckpt)

        kwargs_values = {
            "dataset": "AlpacaDataset",
            "train_on_input": False,
            "seed": 9,
            "shuffle": True,
            "model": ckpt,
            "model_checkpoint": self._fetch_ckpt_model_path(ckpt),
            "tokenizer": "llama2_tokenizer",
            "tokenizer_checkpoint": "/tmp/test-artifacts/tokenizer.model",
            "batch_size": 8,
            "lr": 2e-5,
            "epochs": 2,
            "max_steps_per_epoch": 2,
            "optimizer": "AdamW",
            "loss": "CrossEntropyLoss",
            "output_dir": "/tmp",
            "device": "cpu",
            "dtype": "fp32",
            "enable_activation_checkpointing": False,
            "enable_fsdp": False,
            "run_generation": None,
            "metric_logger_type": "disk",
            "project": None,
            "resume_from_checkpoint": False,
            "cpu_offload": False,
        }

        finetune_llm.recipe(FullFinetuneParams(**kwargs_values))
        loss_values = self._fetch_loss_values(capsys.readouterr().err)
        logger.info("Expected loss values : ", expected_loss_values)
        logger.info("Loss values from Finetune : ", loss_values)
        assert len(loss_values) == len(expected_loss_values)
        for key, value in loss_values.items():
            assert key in expected_loss_values
            expected_loss_value = expected_loss_values[key]
            assert value == pytest.approx(expected_loss_value, abs=0.001)

    def test_finetune_errors(self, capsys, pytestconfig):
        large_scale = pytestconfig.getoption("--large-scale")
        ckpt = "llama2_7b" if large_scale else "small_test_ckpt"
        expected_loss_values = self._fetch_expected_loss_values(ckpt)

        kwargs_values = {
            "dataset": "AlpacaDataset",
            "train_on_input": False,
            "seed": 9,
            "shuffle": True,
            "model": ckpt,
            "model_checkpoint": self._fetch_ckpt_model_path(ckpt),
            "tokenizer": "llama2_tokenizer",
            "tokenizer_checkpoint": "/tmp/test-artifacts/tokenizer.model",
            "batch_size": 8,
            "lr": 2e-5,
            "epochs": 2,
            "max_steps_per_epoch": 2,
            "optimizer": "AdamW",
            "loss": "CrossEntropyLoss",
            "output_dir": "/tmp",
            "device": "cpu",
            "dtype": "fp32",
            "enable_activation_checkpointing": False,
            "enable_fsdp": False,
            "run_generation": None,
            "metric_logger_type": "disk",
            "project": None,
            "resume_from_checkpoint": False,
            "cpu_offload": True,
        }

        with pytest.raises(
            ValueError,
            match="Cannot offload model to CPU if device is not cuda or <= 1 GPUs.",
        ):
            finetune_llm.recipe(FullFinetuneParams(**kwargs_values))


class TestFullFinetuneRecipe:
    def _fetch_loss_values(self, output) -> Dict[str, float]:
        lines = output.splitlines()
        loss_values = {}
        for line in lines:
            if "Loss:" in line:
                splits = line.split("Loss:")
                loss_value = float(splits[1].split(":")[0])
                loss_values[splits[0]] = loss_value
        return loss_values

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

    def _fetch_ckpt_model_path(self, ckpt) -> str:
        if ckpt == "small_test_ckpt":
            return "/tmp/test-artifacts/small-ckpt-01242024"
        if ckpt == "llama2_7b":
            return "/tmp/test-artifacts/llama2-7b-01242024"
        raise ValueError(f"Unknown ckpt {ckpt}")

    def test_loss(self, capsys, pytestconfig):
        large_scale = pytestconfig.getoption("--large-scale")
        ckpt = "llama2_7b" if large_scale else "small_test_ckpt"
        expected_loss_values = self._fetch_expected_loss_values(ckpt)

        kwargs_values = {
            "dataset": "AlpacaDataset",
            "train_on_input": False,
            "seed": 9,
            "shuffle": True,
            "model": ckpt,
            "model_checkpoint": self._fetch_ckpt_model_path(ckpt),
            "tokenizer": "llama2_tokenizer",
            "tokenizer_checkpoint": "/tmp/test-artifacts/tokenizer.model",
            "batch_size": 8,
            "lr": 2e-5,
            "epochs": 2,
            "max_steps_per_epoch": 2,
            "optimizer": "AdamW",
            "loss": "CrossEntropyLoss",
            "output_dir": "/tmp",
            "device": "cpu",
            "dtype": "fp32",
            "resume_from_checkpoint": False,
            "enable_fsdp": False,
            "enable_activation_checkpointing": False,
            "metric_logger_type": "disk",
        }

        recipe_params = FullFinetuneParams(**kwargs_values)

        recipe = FullFinetuneRecipe(recipe_params)
        recipe.setup(params=recipe_params)
        recipe.train()

        loss_values = self._fetch_loss_values(capsys.readouterr().err)
        logger.info("Expected loss values : ", expected_loss_values)
        logger.info("Loss values from Finetune : ", loss_values)
        assert len(loss_values) == len(expected_loss_values)
        for key, value in loss_values.items():
            assert key in expected_loss_values
            expected_loss_value = expected_loss_values[key]
            assert value == pytest.approx(expected_loss_value, abs=0.001)

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
                "model_checkpoint": self._fetch_ckpt_model_path(model_ckpt),
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
