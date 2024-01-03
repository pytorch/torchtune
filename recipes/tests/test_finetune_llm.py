# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict

import pytest
import recipes.finetune_llm as finetune_llm
from torchtune import models

from torchtune.models.llama2.transformer import TransformerDecoder


def small_test_ckpt(vocab_size: int) -> TransformerDecoder:
    return TransformerDecoder(
        vocab_size=32_000,
        num_layers=4,
        num_heads=16,
        embed_dim=256,
        max_seq_len=2048,
        norm_eps=1e-5,
        num_kv_heads=8,
    )


models._MODEL_DICT["small_test_ckpt"] = small_test_ckpt
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

    def test_small_test_ckpt_finetune_loss(self, capsys):
        expected_loss_values = {
            "1|1|": 10.5483,
            "1|2|": 10.5776,
            "2|1|": 10.5696,
            "2|2|": 10.5647,
        }
        kwargs_values = {
            "dataset": "alpaca",
            "dataloader_seed": 9,
            "shuffle": True,
            "model": "small_test_ckpt",
            "model_checkpoint": "/tmp/test-artifacts/small_ckpt.model",
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
            "fsdp": False,
            "activation_checkpointing": False,
        }

        finetune_llm.recipe(kwargs_values)
        loss_values = self._fetch_loss_values(capsys.readouterr().err)
        logger.info("Expected loss values : ", expected_loss_values)
        logger.info("Loss values from Finetune : ", loss_values)
        assert len(loss_values) == len(expected_loss_values)
        for key, value in loss_values.items():
            assert key in expected_loss_values
            expected_loss_value = expected_loss_values[key]
            assert value == pytest.approx(expected_loss_value, abs=0.001)
