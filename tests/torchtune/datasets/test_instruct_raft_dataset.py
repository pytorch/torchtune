# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

from tests.test_utils import DummyTokenizer

from torchtune.datasets import InstructDatasetDeepLakeRAFT


def dummy_transform(sample):
    sample["instruction"] = sample["instruction"] + " asdfghjkl; "
    return sample


class DummyTemplate:
    def __init__(self, template):
        self.template = template

    def format(self, sample, column_map):
        return self.template.format(**sample)


class TestInstructDatasetDeepLakeRAFT:
    template = DummyTemplate("Instruction:\n{instruction}\n\nResponse: ")
    expected_tokenized_prompts = [
        [0, 12, 4, 2, 3, 2, 12, 10, 9, 1, 5, 4, 4, 3, 6, 2, 4, -1]
    ]

    def get_samples(self):
        return [
            {
                "instruction": "This is not an instruction.",
                "cot_answer": "I never know what I'm doing, do you?",
            },
        ]

    @mock.patch("torchtune.datasets._instruct_raft.load_deep_lake_dataset")
    def test_get_item_train(self, mock_load_deep_lake_dataset):
        mock_load_deep_lake_dataset.return_value = self.get_samples()
        expected_labels = self.expected_tokenized_prompts

        dataset = InstructDatasetDeepLakeRAFT(
            tokenizer=DummyTokenizer(),
            source="iam/agoofy/goober",
            template=self.template,
            transform=dummy_transform,
        )
        assert len(dataset) == 1
        mock_load_deep_lake_dataset.assert_called_once()

        prompt, label = dataset[0]
        assert prompt == self.expected_tokenized_prompts[0]
        assert label == expected_labels[0]
