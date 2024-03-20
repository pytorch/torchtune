# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest
from torchtune.data import AlpacaInstructTemplate
from torchtune.datasets._common import CROSS_ENTROPY_IGNORE_IDX

from torchtune.datasets._instruct import _get_template, InstructDataset


class DummyTokenizer:
    def encode(self, text, **kwargs):
        words = text.split()
        return [len(word) for word in words]


def dummy_transform(sample):
    sample["input"] = sample["input"] + " asdfghjkl; "
    sample["instruction"] = sample["instruction"] + " asdfghjkl; "
    return sample


class DummyTemplate:
    def __init__(self, template):
        self.template = template

    def format(self, sample, column_map):
        return self.template.format(**sample)


class TestInstructDataset:
    template = DummyTemplate(
        "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse: "
    )
    expected_tokenized_prompts = [
        [12, 4, 2, 3, 2, 12, 10, 6, 4, 2, 3, 2, 6, 10, 9, 1, 5, 4, 4, 3, 6, 2, 4],
        [12, 4, 2, 2, 12, 10, 6, 4, 2, 2, 6, 10, 9, 1, 6, 4, 4, 3, 6, 2, 4],
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
        mock_load_dataset.return_value = self.get_samples()
        prompt_lengths = (15, 13)
        expected_labels = [
            [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[0] + [1, 5, 4, 4, 3, 6, 2, 4],
            [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[1] + [1, 6, 4, 4, 3, 6, 2, 4],
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
            prompt, label = dataset[i]
            print(prompt, label)
            assert prompt == self.expected_tokenized_prompts[i]
            assert label == expected_labels[i]

    @mock.patch("torchtune.datasets._instruct.load_dataset")
    def test_get_item_train_on_input(self, mock_load_dataset):
        mock_load_dataset.return_value = self.get_samples()
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
            prompt, label = dataset[i]
            assert prompt == self.expected_tokenized_prompts[i]
            assert label == expected_labels[i]


def test_get_template():
    # Test valid template class
    template = _get_template("AlpacaInstructTemplate")
    assert isinstance(template, AlpacaInstructTemplate)

    # Test invalid template class
    with pytest.raises(
        ValueError,
        match="Must be a PromptTemplate class or a string with placeholders.",
    ):
        _ = _get_template("InvalidTemplate")

    # Test valid template strings
    s = [
        "Instruction: {instruction}\nInput: {input}",
        "Instruction: {instruction}",
        "{a}",
    ]
    for t in s:
        assert _get_template(t) == t

    # Test invalid template strings
    s = ["hello", "{}", "a}{b"]
    for t in s:
        with pytest.raises(
            ValueError,
            match="Must be a PromptTemplate class or a string with placeholders.",
        ):
            _ = _get_template(t)
