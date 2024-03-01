# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from unittest.mock import patch

import pytest

from tests.test_utils import get_assets_path

from torchtune.datasets._slimorca import _Llama2ChatFormatConstants, SlimOrcaDataset
from torchtune.modules.tokenizer import Tokenizer


class TestSlimOrcaDataset:
    @pytest.fixture
    def tokenizer(self):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        return Tokenizer.from_file(str(get_assets_path() / "m.model"))

    @patch("torchtune.datasets._slimorca.load_dataset")
    def test_prompt_label_generation(self, load_dataset, tokenizer):
        load_dataset.return_value = []
        dataset = SlimOrcaDataset(tokenizer=tokenizer)
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
            == f"{_Llama2ChatFormatConstants.B_INST} {_Llama2ChatFormatConstants.B_SYS}hi{_Llama2ChatFormatConstants.E_SYS}mid {_Llama2ChatFormatConstants.E_INST}"  # noqa: B950
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
            == f"{_Llama2ChatFormatConstants.B_INST} mid {_Llama2ChatFormatConstants.E_INST}"
        )
        assert label == " lo "

    @patch("torchtune.datasets._slimorca.load_dataset")
    def test_token_generation(self, load_dataset, tokenizer):
        load_dataset.return_value = []
        dataset = SlimOrcaDataset(tokenizer=tokenizer, max_token_length=4096)
        input, label = dataset._generate_tokens("Hello ", "world!")
        assert input == [tokenizer.bos_id, 12, 1803, 1024, 103, tokenizer.eos_id]
        assert label == ([-100] * 3 + [1024, 103, tokenizer.eos_id])

    @patch("torchtune.datasets._slimorca.load_dataset")
    def test_truncated_token_generation(self, load_dataset, tokenizer):
        load_dataset.return_value = []
        dataset = SlimOrcaDataset(tokenizer=tokenizer, max_token_length=5)
        # 5 is enough for full prompt, but not for label
        input, label = dataset._generate_tokens("Hello ", "world!")
        assert input == [tokenizer.bos_id, 12, 1803, 1024, tokenizer.eos_id]
        assert label == ([-100] * 3 + [1024, tokenizer.eos_id])

        # 4 is not enough for full prompt nor response but truncation
        # is still feasible
        dataset = SlimOrcaDataset(tokenizer=tokenizer, max_token_length=4)
        input, label = dataset._generate_tokens("Hello ", "world!")
        assert input == [tokenizer.bos_id, 12, 1024, tokenizer.eos_id]
        assert label == ([-100] * 2 + [1024, tokenizer.eos_id])

    @patch("torchtune.datasets._slimorca.load_dataset")
    def test_value_error(self, load_dataset, tokenizer):
        load_dataset.return_value = []
        with pytest.raises(ValueError):
            SlimOrcaDataset(tokenizer=tokenizer, max_token_length=3)

    @patch("torchtune.datasets._slimorca.load_dataset")
    @pytest.mark.parametrize("max_token_length", [128, 512, 1024, 4096])
    def test_dataset_get_item(self, load_dataset, tokenizer, max_token_length):
        # Sample data from slimorca dataset
        load_dataset.return_value = [
            {
                "conversations": [
                    {
                        "from": "system",
                        "value": "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",  # noqa: B950
                    },
                    {
                        "from": "human",
                        "value": "Please briefly summarize this news article:\n\nAOL.com Video - Father Lets 8-Year-Old Drive On Icy Road\n\nDescription:Would you let your 8-year-old drive your car? How about on an icy road? Well one father in Russia did just that, and recorded the entire thing. To her credit, the child seemed to be doing a great job. (0:44)\n\nTags: 8-year-old driver , caught on camera , child driver , pix11\n\nSummary:",  # noqa: B950
                    },
                    {
                        "from": "gpt",
                        "value": "A father in Russia allowed his 8-year-old child to drive his car on an icy road and recorded the event. The child appeared to be handling the situation well, showcasing their driving skills despite the challenging conditions.",  # noqa: B950
                    },
                ]
            }
        ]
        ds = SlimOrcaDataset(tokenizer=tokenizer, max_token_length=max_token_length)
        input, label = ds[0]
        assert len(input) <= max_token_length
        assert len(label) <= max_token_length
        assert len(input) == len(label)
        assert input[0] == tokenizer.bos_id
        assert input[-1] == tokenizer.eos_id
        assert label[-1] == tokenizer.eos_id
