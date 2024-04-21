# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
from datasets import Dataset

from tests.test_utils import get_assets_path
from torchtune.datasets import stack_dataset
from torchtune.modules.tokenizers import SentencePieceTokenizer


class TestAlpacaDataset:
    @pytest.fixture
    def tokenizer(self):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        return SentencePieceTokenizer(str(get_assets_path() / "m.model"))

    def get_samples(self):
        samples_list = [
            "This is a packing test",
            "A fantastic test. It should pack two samples.",
            "This one will not be fully packed.",
        ]

        samples_dict = {"content": samples_list}

        return Dataset.from_dict(samples_dict)

    @patch("torchtune.datasets._concat.load_dataset")
    def test_label(self, load_dataset, tokenizer):
        """
        Test whether the input and the labels are correctly created.
        """

        # mock the call to HF datasets
        load_dataset.return_value = self.get_samples()

        stack_ds = stack_dataset(
            tokenizer=tokenizer,
            max_seq_len=10,
            train_split_name=None,
        )

        for i in range(len(stack_ds)):
            inputs, label = stack_ds[i]
            assert len(inputs) == 10
            assert len(inputs) == len(label)
            assert inputs == label

            if i == 0:
                assert inputs[0] == tokenizer.bos_id
