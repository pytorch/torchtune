# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

from tests.test_utils import DummyTokenizer

from torchtune.datasets import TextCompletionDataset


class TestTextCompletionDataset:
    expected_tokenized_prompts = [
        [0, 4, 2, 2, 7, 5, -1],
        [0, 4, 2, 7, 7, 5, -1],
    ]

    def get_samples(self):
        return [
            {
                "text": "This is an example text.",
            },
            {
                "text": "This is another example text.",
            },
        ]

    @mock.patch("torchtune.datasets._text_completion.load_dataset")
    def test_get_item(self, mock_load_dataset):
        mock_load_dataset.return_value = self.get_samples()
        expected_labels = self.expected_tokenized_prompts

        dataset = TextCompletionDataset(
            tokenizer=DummyTokenizer(),
            source="iam/agoofy/goober",
            column="text",
            max_seq_len=100,
        )
        assert len(dataset) == 2
        mock_load_dataset.assert_called_once()

        for i in range(len(dataset)):
            prompt, label = dataset[i]["tokens"], dataset[i]["labels"]
            assert prompt == self.expected_tokenized_prompts[i]
            assert label == expected_labels[i]

    @mock.patch("torchtune.datasets._text_completion.load_dataset")
    def test_get_item_no_eos(self, mock_load_dataset):
        mock_load_dataset.return_value = self.get_samples()
        expected_labels = self.expected_tokenized_prompts

        dataset = TextCompletionDataset(
            tokenizer=DummyTokenizer(),
            source="iam/agoofy/goober",
            column="text",
            max_seq_len=100,
            add_eos=False,
        )
        assert len(dataset) == 2
        mock_load_dataset.assert_called_once()

        for i in range(len(dataset)):
            prompt, label = dataset[i]["tokens"], dataset[i]["labels"]
            # trimming EOS IDs from the expected tokens, assertion is against:
            # [0, 4, 2, 2, 7, 5]
            assert prompt == self.expected_tokenized_prompts[i][:-1]
            assert label == expected_labels[i][:-1]
