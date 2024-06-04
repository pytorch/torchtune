# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

from datasets import Dataset
from tests.test_utils import DummyTokenizer
from torchtune.data import InstructTemplate
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.datasets import InstructDataset


def dummy_transform(sample):
    sample["input"] = sample["input"] + " asdfghjkl; "
    sample["instruction"] = sample["instruction"] + " asdfghjkl; "
    return sample


class DummyTemplate(InstructTemplate):
    template = "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse: "

    @classmethod
    def format(cls, sample, column_map):
        return cls.template.format(**sample)


class TestInstructDataset:
    template = DummyTemplate
    expected_tokenized_prompts = [
        [
            0,
            12,
            4,
            2,
            3,
            2,
            12,
            10,
            6,
            4,
            2,
            3,
            2,
            6,
            10,
            9,
            1,
            5,
            4,
            4,
            3,
            6,
            2,
            4,
            -1,
        ],
        [0, 12, 4, 2, 2, 12, 10, 6, 4, 2, 2, 6, 10, 9, 1, 6, 4, 4, 3, 6, 2, 4, -1],
        [
            0,
            12,
            4,
            2,
            3,
            2,
            12,
            10,
            6,
            4,
            2,
            3,
            2,
            6,
            10,
            9,
            1,
            5,
            4,
            4,
            3,
            6,
            2,
            4,
            -1,
        ],
        [0, 12, 4, 2, 2, 12, 10, 6, 4, 2, 2, 6, 10, 9, 1, 6, 4, 4, 3, 6, 2, 4, -1],
    ]

    def get_samples(self):
        return [
            {
                "instruction": "This is not an instruction.",
                "input": "This is not an input.",
                "output": "I never know what I'm doing, do you?",
            },
            {
                "instruction": "This is an instruction.",
                "input": "This is an input.",
                "output": "I always know what I'm doing, do you?",
            },
        ]

    @mock.patch("torchtune.datasets._instruct.load_dataset")
    def test_get_item_no_train_on_input(self, mock_load_dataset):
        mock_load_dataset.return_value = Dataset.from_list(self.get_samples())
        prompt_lengths = (16, 14)
        expected_labels = [
            [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[0]
            + [1, 5, 4, 4, 3, 6, 2, 4, -1],
            [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[1]
            + [1, 6, 4, 4, 3, 6, 2, 4, -1],
        ]

        dataset = InstructDataset(
            tokenizer=DummyTokenizer(),
            source="iam/agoofy/goober",
            template=self.template,
            transform=dummy_transform,
            train_on_input=False,
        )
        assert len(dataset) == 2
        mock_load_dataset.assert_called_once()

        for i in range(len(dataset)):
            prompt, label = dataset[i]["tokens"], dataset[i]["labels"]
            assert prompt == self.expected_tokenized_prompts[i]
            assert label == expected_labels[i]

    @mock.patch("torchtune.datasets._instruct.load_dataset")
    def test_get_item_train_on_input(self, mock_load_dataset):
        mock_load_dataset.return_value = Dataset.from_list(self.get_samples())
        expected_labels = self.expected_tokenized_prompts

        dataset = InstructDataset(
            tokenizer=DummyTokenizer(),
            source="iam/agoofy/goober",
            template=self.template,
            transform=dummy_transform,
            train_on_input=True,
        )
        assert len(dataset) == 2
        mock_load_dataset.assert_called_once()

        for i in range(len(dataset)):
            prompt, label = dataset[i]["tokens"], dataset[i]["labels"]
            assert prompt == self.expected_tokenized_prompts[i]
            assert label == expected_labels[i]
