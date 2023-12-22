# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import pytest
import recipes.finetune_llm as finetune_llm


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
            "1|1|": 12.5535,
            "1|2|": 8.7051,
            "2|1|": 8.0128,
            "2|2|": 7.4046,
        }
        argv_values = [
            "--dataset",
            "alpaca",
            "--dataloader-seed",
            "9",
            "--model",
            "llama2_7b",
            "--model-checkpoint",
            "test-artifacts/llama2-7b-native-checkpoint",
            "--tokenizer",
            "llama2_tokenizer",
            "--tokenizer-checkpoint",
            "test-artifacts/tokenizer.model",
            "--batch-size",
            "8",
            "--num-batches",
            "2",
            "--epochs",
            "2",
            "--device",
            "cpu",
        ]

        finetune_llm.main(argv_values)
        loss_values = self._fetch_loss_values(capsys.readouterr().err)
        print("Loss values from Finetune : ", loss_values)
        assert len(loss_values) == len(expected_loss_values)
        for key, value in loss_values.items():
            assert key in expected_loss_values
            expected_loss_value = expected_loss_values[key]
            assert value == pytest.approx(expected_loss_value, abs=0.001)
