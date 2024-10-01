# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import pytest
import torch
from tests.test_utils import DummyTokenizer
from torch.utils.data import Dataset

from torchtune.datasets import PackedDataset


class DummyDataset(Dataset):
    def __init__(self, sample_size):
        self.sample_size = sample_size

    def __getitem__(self, index):
        if index >= 1000:
            raise IndexError()
        return {
            "tokens": [index] * self.sample_size,
            "labels": [index] * self.sample_size,
        }

    def __len__(self):
        return 1000


class DummyRealDataset(Dataset):
    def __init__(self):
        self.samples_list = [
            "This is a packing test",
            "A fantastic test. It should pack two samples.",
            "This one will not be fully packed.",
        ]
        self.tokenizer = DummyTokenizer()

    def __getitem__(self, index):
        tokens = self.tokenizer.encode(self.samples_list[index])
        return {"tokens": tokens, "labels": tokens}

    def __len__(self):
        return len(self.samples_list)


class TestPackedDataset:
    def _get_expected_seq_lens_and_input_pos(
        self, max_seq_len, sample_size, split_across_pack
    ):
        """
        Generate expected seq lens and position ids for given max sequence
        length and sample length
        """
        num_samples, remainder = divmod(max_seq_len, sample_size)
        seq_lens = [sample_size] * num_samples
        if split_across_pack and remainder > 0:
            num_samples += 1
        input_pos = [list(range(sample_size)) for i in range(1, num_samples + 1)]
        input_pos = list(itertools.chain(*input_pos))

        # Emulate seq len and position id padding
        if remainder > 0:
            if not split_across_pack:
                input_pos.extend(list(range(sample_size, sample_size + remainder)))
            seq_lens.extend([remainder])

        return torch.tensor(seq_lens), torch.tensor(input_pos[:max_seq_len])

    def _calculate_num_packs(
        self, dataset_size, max_seq_len, sample_size, split_across_pack, max_packs
    ):
        # First see how many samples we can fit in a single pack
        num_samples_per_pack, remainder = divmod(max_seq_len, sample_size)

        # If we split across pack (and the samples don't fit perfectly in max_seq_len), we can fit more
        if split_across_pack and remainder > 0:
            # Now we need the fractional to see how many we can partially fit in each pack
            num_samples_per_pack = max_seq_len / sample_size

        # If we don't split across pack, we will need more packs
        num_packs, remainder = divmod(dataset_size, num_samples_per_pack)

        # If there's leftover, we need to add one more pack
        if remainder > 0:
            num_packs += 1

        return num_packs if num_packs < max_packs else max_packs

    @pytest.mark.parametrize("max_seq_len", [25])
    @pytest.mark.parametrize("sample_size", [2, 5])
    @pytest.mark.parametrize("max_packs", [5, 200])
    @pytest.mark.parametrize("split_across_pack", [True, False])
    def test_packed_dataset(
        self, max_seq_len, sample_size, max_packs, split_across_pack
    ):
        dataset = DummyDataset(sample_size)
        packed = PackedDataset(
            dataset,
            max_seq_len=max_seq_len,
            max_packs=max_packs,
            split_across_pack=split_across_pack,
        )

        # Check we get right number of packs
        correct_num_packs = self._calculate_num_packs(
            len(dataset), max_seq_len, sample_size, split_across_pack, max_packs
        )
        assert len(packed) == correct_num_packs

        # Check all fields are same length
        assert (
            len(packed[0]["tokens"])
            == len(packed[0]["labels"])
            == len(packed[0]["input_pos"])
        )
        # Check that samples are packed correctly - very last individual sample
        # should have index value of the number of times dataset was iterated over
        if split_across_pack:
            # If we split samples, we'll know how many samples by taking the
            # full length and dividing by sample size
            last_index, remainder = divmod(len(packed) * max_seq_len, sample_size)
            # Account for remaining sample that didn't fit in window
            last_index = last_index if remainder > 0 else last_index - 1
        else:
            # If we don't split samples, we know how many samples by taking
            # how much fits in a single window and multiplying by max rows.
            # If there is a remainder, this will end up being a pad token.
            last_index = (
                (max_seq_len // sample_size) * len(packed) - 1
                if max_seq_len % sample_size == 0
                else 0
            )

        assert packed[-1]["tokens"][-1].item() == last_index

        (
            expected_seq_lens,
            expected_input_pos,
        ) = self._get_expected_seq_lens_and_input_pos(
            max_seq_len, sample_size, split_across_pack
        )

        torch.testing.assert_close(packed[0]["seq_lens"], expected_seq_lens)
        torch.testing.assert_close(packed[0]["input_pos"], expected_input_pos)

    @pytest.mark.parametrize("max_seq_len", [13])
    @pytest.mark.parametrize("sample_size", [14, 27, 40])
    @pytest.mark.parametrize("max_packs", [5, 200, 3100])
    @pytest.mark.parametrize("split_across_pack", [True])
    def test_chunked_case(self, max_seq_len, sample_size, max_packs, split_across_pack):
        dataset = DummyDataset(sample_size)
        packed = PackedDataset(
            dataset,
            max_seq_len=max_seq_len,
            max_packs=max_packs,
            split_across_pack=split_across_pack,
        )

        # Check we get right number of packs
        correct_num_packs = self._calculate_num_packs(
            len(dataset), max_seq_len, sample_size, split_across_pack, max_packs
        )
        assert len(packed) == correct_num_packs

        # Check all fields are same length
        assert all(
            len(pack["tokens"]) == len(pack["labels"]) == len(pack["input_pos"])
            for pack in packed
        )

        # Check that all sum(seq_lens) are equal to max_seq_len
        assert all(pack["seq_lens"].sum().item() == max_seq_len for pack in packed)

    def test_packed_dataset_real_data(self):
        expected_tokenized_prompts = [
            torch.tensor([0, 4, 2, 1, 7, 4, -1, 0, 1, 9]),
            torch.tensor([5, 2, 6, 4, 3, 8, -1, 0, 4, 3]),
            torch.tensor([4, 3, 2, 5, 7, -1, 0, 0, 0, 0]),
        ]
        expected_tokenized_labels = [
            torch.tensor([0, 4, 2, 1, 7, 4, -1, 0, 1, 9]),
            torch.tensor([5, 2, 6, 4, 3, 8, -1, 0, 4, 3]),
            torch.tensor([4, 3, 2, 5, 7, -1, -100, -100, -100, -100]),
        ]
        expected_seq_lens = [
            torch.tensor(
                [7, 3],
            ),
            torch.tensor(
                [7, 3],
            ),
            torch.tensor(
                [6, 4],
            ),
        ]
        expected_input_pos = [
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2]),
            torch.tensor([3, 4, 5, 6, 7, 8, 9, 0, 1, 2]),
            # Padded position ids cannot go beyond max seq_len - 1
            torch.tensor([3, 4, 5, 6, 7, 8, 9, 9, 9, 9]),
        ]
        packed = PackedDataset(
            DummyRealDataset(),
            max_seq_len=10,
            split_across_pack=True,
        )

        for i in range(len(packed)):
            prompt, label, seq_lens, input_pos = (
                packed[i]["tokens"],
                packed[i]["labels"],
                packed[i]["seq_lens"],
                packed[i]["input_pos"],
            )
            torch.testing.assert_close(prompt, expected_tokenized_prompts[i])
            torch.testing.assert_close(label, expected_tokenized_labels[i])
            torch.testing.assert_close(input_pos, expected_input_pos[i])
            torch.testing.assert_close(seq_lens, expected_seq_lens[i])

    def test_pad_pack(self):
        padding_idx = -8
        ignore_idx = -100  # Same as CROSS_ENTROPY_IGNORE_IDX
        pack = {
            "tokens": [2, 5],
            "labels": [3, 7],
            "seq_lens": [1, 1],
            # Let the first token be the end of the previous sample (pos 8),
            # and the second token the start of the next sample (pos 0). Collate
            # should continue from 0 -> 1, 2, ...
            "input_pos": [8, 0],
        }

        dataset = DummyDataset(2)
        packed = PackedDataset(
            dataset,
            max_seq_len=4,
        )

        pack = packed._convert_to_tensors(pack)
        padded = packed._pad_pack(pack, padding_idx=padding_idx)

        padded_input = padded["tokens"]
        padded_label = padded["labels"]
        padded_input_pos = padded["input_pos"]
        padded_seq_lens = padded["seq_lens"]

        torch.testing.assert_close(
            padded_input, torch.tensor([2, 5, padding_idx, padding_idx])
        )
        torch.testing.assert_close(
            padded_label, torch.tensor([3, 7, ignore_idx, ignore_idx])
        )
        torch.testing.assert_close(padded_input_pos, torch.tensor([8, 0, 1, 2]))
        torch.testing.assert_close(padded_seq_lens, torch.tensor([1, 1, 2]))

    def test_pack_errors_if_sample_too_long(self):
        dataset = DummyDataset(8)
        with pytest.raises(ValueError, match="Dataset sample is too long"):
            PackedDataset(
                dataset,
                max_seq_len=4,
            )
