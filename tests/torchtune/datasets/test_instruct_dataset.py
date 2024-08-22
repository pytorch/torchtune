# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tests.common import ASSETS
from tests.test_utils import DummyTokenizer
from torchtune.data import InstructTemplate
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.datasets import instruct_dataset, InstructDataset


def dummy_transform(sample):
    sample["instruction"] = sample["instruction"] + " asdfghjkl; "
    sample["response"] = sample["response"] + " asdfghjkl; "
    return sample


class DummyTemplate(InstructTemplate):
    template = "Instruction:\n{instruction}\n\nResponse:\n"

    @classmethod
    def format(cls, sample, column_map):
        return cls.template.format(**sample)


class TestInstructDataset:
    @pytest.mark.parametrize("train_on_input", [True, False])
    def test_get_item(self, train_on_input):
        template = DummyTemplate
        expected_tokenized_prompts = [
            [0, 12, 4, 4, 2, 2, 2, 7, 10, 9, 2, 2, 5, 2, 2, 6, 10, -1],
            [0, 12, 2, 2, 8, 2, 15, 10, 9, 8, 3, 15, 3, 4, 9, 3, 15, 10, -1],
        ]
        prompt_lengths = (10, 9)
        expected_labels = [
            [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[0] + [2, 2, 5, 2, 2, 6, 10, -1],
            [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[1]
            + [8, 3, 15, 3, 4, 9, 3, 15, 10, -1],
        ]

        dataset = InstructDataset(
            tokenizer=DummyTokenizer(),
            source="json",
            template=template,
            transform=dummy_transform,
            train_on_input=train_on_input,
            data_files=str(ASSETS / "instruct_tiny.json"),
            column_map={"output": "response"},
            split="train",
        )
        assert len(dataset) == 2

        for i in range(len(dataset)):
            prompt, label = dataset[i]["tokens"], dataset[i]["labels"]
            assert prompt == expected_tokenized_prompts[i]
            if train_on_input:
                assert label == expected_tokenized_prompts[i]
            else:
                assert label == expected_labels[i]

        expected_tokenized_prompts = [
            [0, 4, 4, 2, 2, 2, 7, 2, 2, 5, 2, 2, 6, -1],
            [0, 2, 2, 8, 2, 15, 8, 3, 15, 3, 4, 9, 3, 15, -1],
        ]
        prompt_lengths = (7, 6)
        expected_labels = [
            [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[0] + [2, 2, 5, 2, 2, 6, -1],
            [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[1]
            + [8, 3, 15, 3, 4, 9, 3, 15, -1],
        ]

        dataset = instruct_dataset(
            tokenizer=DummyTokenizer(),
            source="json",
            train_on_input=train_on_input,
            data_files=str(ASSETS / "instruct_tiny.json"),
            column_map={"input": "instruction", "output": "response"},
            split="train",
        )
        assert len(dataset) == 2

        for i in range(len(dataset)):
            prompt, label = dataset[i]["tokens"], dataset[i]["labels"]
            assert prompt == expected_tokenized_prompts[i]
            if train_on_input:
                assert label == expected_tokenized_prompts[i]
            else:
                assert label == expected_labels[i]
