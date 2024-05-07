# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest

from tests.test_utils import get_assets_path
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.datasets import raft_dataset
from torchtune.modules.tokenizers import SentencePieceTokenizer


class TestRAFTDataset:
    @pytest.fixture
    def tokenizer(self):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        return SentencePieceTokenizer(str(get_assets_path() / "m.model"))

    @patch("torchtune.datasets._instruct_raft.load_deep_lake_dataset")
    def test_label_no_masking(self, load_deep_lake_dataset, tokenizer):
        """
        Test whether the input and the labels are correctly created when the input is not masked.
        """

        # mock the call to Deep Lake Datasets
        load_deep_lake_dataset.return_value = [
            {
                "instruction": """<DOCUMENT> Artificial Intelligence (AI) is revolutionizing industries worldwide,
                from healthcare to finance.By analyzing vast amounts of data, AI algorithms can detect patterns and
                make predictions with unprecedented accuracy.</DOCUMENT><DOCUMENT>In the realm of autonomous vehicles,
                AI plays a pivotal role in enabling safe and efficient navigation.
                Through real-time data processing and machine learning, self-driving cars can adapt to diverse road
                conditions and make split-second decisions.</DOCUMENT>
                <DOCUMENT>AI-powered virtual assistants, like Siri and Alexa, have become ubiquitous in our daily lives.
                These intelligent systems utilize natural language processing and machine learning algorithms
                to understand and respond to user queries,simplifying tasks and enhancing user experience.</DOCUMENT>
                What are some key applications of artificial intelligence across different industries?""",
                "cot_answer": """##Reason: To answer the question about the applications of artificial intelligence
                across different industries, we need to consider the provided context.
                Here is the step-by-step reasoning:
                Identify key examples of AI applications mentioned in the provided texts:
                Healthcare: AI is used for data analysis, pattern recognition, and predictive modeling.
                Autonomous vehicles: AI enables safe navigation and decision-making in self-driving cars.
                Virtual assistants: AI powers intelligent systems like Siri and Alexa for natural language
                processing and task automation.
                Summarize the main industries benefiting from AI: Healthcare, transportation, and consumer electronics.
                ##Answer: Some key applications of artificial intelligence across different industries
                include healthcare analytics, autonomous vehicles for transportation,and virtual assistants
                in consumer electronics.""",
            }
        ]

        raft_ds = raft_dataset(tokenizer=tokenizer)
        input, labels = raft_ds[0]

        assert len(input) == len(labels)
        assert labels[-1] == tokenizer.eos_id
        assert input[0] == tokenizer.bos_id
        assert CROSS_ENTROPY_IGNORE_IDX not in labels
