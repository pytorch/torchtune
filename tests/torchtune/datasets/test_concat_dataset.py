# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

from datasets import Dataset
from tests.test_utils import DummyTokenizer
from torchtune.datasets import ConcatDataset


class TestInstructDataset:
    expected_tokenized_prompts = [
        [0, 4, 2, 1, 7, 4, -1, 0, 1, 9],
        [5, 2, 6, 4, 3, 8, -1, 0, 4, 3],
    ]

    def get_samples(self):
        samples_list = [
            "This is a packing test",
            "A fantastic test. It should pack two samples.",
            "This one will not be fully packed.",
        ]

        samples_dict = {"content": samples_list}

        return Dataset.from_dict(samples_dict)

    @mock.patch("torchtune.datasets._concat.load_dataset")
    def test_get_item(self, mock_load_dataset):
        mock_load_dataset.return_value = self.get_samples()
        dataset = ConcatDataset(
            tokenizer=DummyTokenizer(),
            source="were/going/jellyfishing",
            text_column="content",
            max_seq_len=10,
        )
        assert len(dataset) == 2
        mock_load_dataset.assert_called_once()

        for i in range(len(dataset)):
            prompt, label = dataset[i]
            assert prompt == self.expected_tokenized_prompts[i]
            assert label == self.expected_tokenized_prompts[i]
