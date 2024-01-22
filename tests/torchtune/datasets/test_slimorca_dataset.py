# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import random
from pathlib import Path

import pytest

from torchtune import datasets
from torchtune.modules.tokenizer import Tokenizer

ASSETS = Path(__file__).parent.parent.parent / "assets"


class TestSlimOrcaDataset:
    @pytest.fixture
    def tokenizer(self):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        return Tokenizer.from_file(str(ASSETS / "m.model"))

    def test_slim_orca_dataset(self, tokenizer):
        dataset = datasets.get_dataset("slimorca", tokenizer=tokenizer)
        assert len(dataset) == 363_491

    def test_prompt_label_generation(self, tokenizer):
        dataset = datasets.get_dataset("slimorca", tokenizer=tokenizer)
        sample = [
            {
                "from": "system",
                "value": "hi",
            },
            {
                "from": "human",
                "value": "mid",
            },
            {
                "from": "gpt",
                "value": "lo",
            },
        ]
        prompt, label = dataset._generate_prompt_label(sample)
        assert (
            prompt
            == f"{datasets.Llama2ChatFormatConstants.B_INST} {datasets.Llama2ChatFormatConstants.B_SYS}hi{datasets.Llama2ChatFormatConstants.E_SYS}mid {datasets.Llama2ChatFormatConstants.E_INST}"  # noqa: B950
        )
        assert label == " lo "

        sample = [
            {
                "from": "human",
                "value": "mid",
            },
            {
                "from": "gpt",
                "value": "lo",
            },
        ]
        prompt, label = dataset._generate_prompt_label(sample)
        assert (
            prompt
            == f"{datasets.Llama2ChatFormatConstants.B_INST} mid {datasets.Llama2ChatFormatConstants.E_INST}"
        )
        assert label == " lo "

    def test_token_generation(self, tokenizer):
        dataset = datasets.get_dataset(
            "slimorca", tokenizer=tokenizer, max_token_length=4096
        )
        input, label = dataset._generate_tokens("Hello ", "world!")
        assert input == [tokenizer.bos_id, 12, 1803, 1024, 103, tokenizer.eos_id]
        assert label == ([-100] * 3 + [1024, 103, tokenizer.eos_id])

    def test_truncated_token_generation(self, tokenizer):
        dataset = datasets.get_dataset(
            "slimorca", tokenizer=tokenizer, max_token_length=5
        )
        # 5 is enough for full prompt, but not for label
        input, label = dataset._generate_tokens("Hello ", "world!")
        assert input == [tokenizer.bos_id, 12, 1803, 1024, tokenizer.eos_id]
        assert label == ([-100] * 3 + [1024, tokenizer.eos_id])

        # 4 is not enough for full prompt nor response but truncation
        # is still feasible
        dataset = datasets.get_dataset(
            "slimorca", tokenizer=tokenizer, max_token_length=4
        )
        input, label = dataset._generate_tokens("Hello ", "world!")
        assert input == [tokenizer.bos_id, 12, 1024, tokenizer.eos_id]
        assert label == ([-100] * 2 + [1024, tokenizer.eos_id])

    def test_value_error(self, tokenizer):
        with pytest.raises(ValueError):
            datasets.get_dataset("slimorca", tokenizer=tokenizer, max_token_length=3)

    @pytest.mark.parametrize("max_token_length", [128, 512, 1024, 4096])
    def test_dataset_get_item(self, tokenizer, max_token_length):
        ds = datasets.get_dataset(
            "slimorca", tokenizer=tokenizer, max_token_length=max_token_length
        )
        index = random.randint(0, len(ds))
        input, label = ds[index]
        assert (
            len(input) <= max_token_length
        ), f"{index} in slimorca fails input token length check"
        assert (
            len(label) <= max_token_length
        ), f"{index} in slimorca fails label token length check"
        assert len(input) == len(
            label
        ), f"{index} in slimorca fails token lists equality check"
        assert input[0] == tokenizer.bos_id, f"{index} in slimorca fails bos check"
        assert input[-1] == tokenizer.eos_id, f"{index} in slimorca fails eos check"
        assert (
            label[-1] == tokenizer.eos_id
        ), f"{index} in slimorca fails label eos check"
