# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial

import pytest

from tests.test_utils import DummyTokenizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune.data import padded_collate_rl
from torchtune.datasets._math import math_dataset


class TestMATHDataset:
    @pytest.fixture
    def tokenizer(self):
        return DummyTokenizer()

    # @pytest.fixture
    # def sample(self):
    #     return {
    #         "instruction": "Give three tips for staying healthy.",
    #         "input": "",
    #         "output": (
    #             "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables."
    #             "2. Exercise regularly to keep your body active and strong."
    #             "3. Get enough sleep and maintain a consistent sleep schedule."
    #         ),
    #     }

    def test_dataset(self, tokenizer):
        ds = math_dataset(tokenizer)
        collate_fn = padded_collate_rl

        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=False,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            batch_size=4,
            sampler=sampler,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=0,
                    ignore_idx=-100,
                )
            ),
        )

        for idx, batch in enumerate(dataloader):
            print(idx, batch)
            break

    # @patch("torchtune.datasets._sft.load_dataset")
    # def test_label_no_masking(self, load_dataset, tokenizer, sample):
    #     """
    #     Test whether the input and the labels are correctly created when the input is not masked.
    #     """
    #
    #     # mock the call to HF datasets
    #     load_dataset.return_value = Dataset.from_list([sample])
    #
    #     alpaca_ds = alpaca_dataset(tokenizer=tokenizer)
    #     input, labels = alpaca_ds[0]["tokens"], alpaca_ds[0]["labels"]
    #
    #     assert len(input) == len(labels)
    #     assert labels[-1] == tokenizer.eos_id
    #     assert input[0] == tokenizer.bos_id
    #     assert CROSS_ENTROPY_IGNORE_IDX not in labels
    #
    # @patch("torchtune.datasets._sft.load_dataset")
    # def test_label_masking(self, load_dataset, tokenizer, sample):
    #     """
    #     Test whether the input and the labels are correctly created when the input is masked.
    #     """
    #
    #     # mock the call to HF datasets
    #     load_dataset.return_value = Dataset.from_list([sample])
    #
    #     alpaca_ds = alpaca_dataset(tokenizer=tokenizer, train_on_input=False)
    #
    #     # Generate the input and labels
    #     input, labels = alpaca_ds[0]["tokens"], alpaca_ds[0]["labels"]
    #
    #     assert len(input) == len(labels)
    #     assert labels[-1] == tokenizer.eos_id
    #     assert input[0] == tokenizer.bos_id
    #     assert labels.count(CROSS_ENTROPY_IGNORE_IDX) == 27
    #
    # @patch("torchtune.datasets._sft.load_dataset")
    # def test_alpaca_clean(self, load_dataset, tokenizer, sample):
    #     """
    #     Test whether the input and the labels are correctly created when the input is not masked.
    #     """
    #
    #     # mock the call to HF datasets
    #     load_dataset.return_value = Dataset.from_list([sample])
    #
    #     alpaca_ds = alpaca_cleaned_dataset(tokenizer=tokenizer)
    #     input, labels = alpaca_ds[0]["tokens"], alpaca_ds[0]["labels"]
    #
    #     assert len(input) == len(labels)
    #     assert labels[-1] == tokenizer.eos_id
    #     assert input[0] == tokenizer.bos_id
    #     assert CROSS_ENTROPY_IGNORE_IDX not in labels


# class TestAlpacaToMessages:
#     @pytest.fixture
#     def sample(self):
#         return {
#             "maybe_instruction": "hello",
#             "maybe_input": "world",
#             "maybe_output": "hello world",
#         }
#
#     @pytest.fixture
#     def sample_no_input(self):
#         return {
#             "maybe_instruction": "hello world",
#             "maybe_input": "",
#             "maybe_output": "hello world",
#         }
#
#     @pytest.mark.parametrize("train_on_input", [True, False])
#     def test_call(self, train_on_input, sample):
#         transform = AlpacaToMessages(
#             column_map={
#                 "instruction": "maybe_instruction",
#                 "input": "maybe_input",
#                 "output": "maybe_output",
#             },
#             train_on_input=train_on_input,
#         )
#         actual = transform(sample)
#         expected = [
#             Message(
#                 role="user",
#                 content="Below is an instruction that describes a task, paired with an input that provides further context. "
#                 "Write a response that appropriately completes the request.\n\n"
#                 "### Instruction:\nhello\n\n### Input:\nworld\n\n### Response:\n",
#                 masked=True,
#                 eot=True,
#             ),
#             Message(role="assistant", content="hello world", masked=False, eot=True),
#         ]
#         assert_dialogue_equal(actual["messages"], expected)
#
#     @pytest.mark.parametrize("train_on_input", [True, False])
#     def test_call_no_input(self, train_on_input, sample_no_input):
#         transform = AlpacaToMessages(
#             column_map={
#                 "instruction": "maybe_instruction",
#                 "input": "maybe_input",
#                 "output": "maybe_output",
#             },
#             train_on_input=train_on_input,
#         )
#         actual = transform(sample_no_input)
#         expected = [
#             Message(
#                 role="user",
#                 content="Below is an instruction that describes a task. "
#                 "Write a response that appropriately completes the request.\n\n"
#                 "### Instruction:\nhello world\n\n### Response:\n",
#                 masked=True,
#                 eot=True,
#             ),
#             Message(role="assistant", content="hello world", masked=False, eot=True),
#         ]
#         assert_dialogue_equal(actual["messages"], expected)
