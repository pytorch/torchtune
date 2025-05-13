# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import time
from collections import deque

import torch
from torch.utils.data import IterableDataset


class OnTheFlyPackedDataset(IterableDataset):
    def __init__(
        self,
        dataset,
        max_seq_len: int,
        padding_idx: int = -1,
        buffer_size: int = 10,
        num_bins: int = 50,
    ):
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.buffer_size = buffer_size
        self.num_bins = num_bins

    def __iter__(self):
        return EfficientPackedIterator(
            self.dataset,
            self.max_seq_len,
            self.padding_idx,
            self.buffer_size,
            self.num_bins,
        )


class EfficientPackedIterator:
    def __init__(self, dataset, max_seq_len, padding_idx, buffer_size, num_bins):
        self.iterator = iter(dataset)
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.buffer_size = buffer_size
        self.num_bins = num_bins
        self.bin_size = max_seq_len // num_bins
        self.bins = [
            (i * self.bin_size, min((i + 1) * self.bin_size, max_seq_len), deque())
            for i in range(num_bins)
        ]
        self.total_buffered = 0
        self.exhausted = False
        self.current_pack_number = 0
        self.ages = []
        self.sequence_id_counter = 0
        self.pack_count = 0
        self.total_padding_count = 0
        self.total_tokens_count = 0
        self.start_time = time.time()
        self._fill_buffer()

    def _fill_buffer(self):
        """Fill bins up to buffer_size with sequences."""
        while self.total_buffered < self.buffer_size and not self.exhausted:
            try:
                seq = next(self.iterator)
                seq_len = len(seq["tokens"])
                if seq_len <= self.max_seq_len:
                    seq["id"] = self.sequence_id_counter
                    seq["added_pack"] = self.current_pack_number
                    self.sequence_id_counter += 1
                    bin_idx = min((seq_len - 1) // self.bin_size, self.num_bins - 1)
                    self.bins[bin_idx][2].append(seq)
                    self.total_buffered += 1
                else:
                    print(
                        f"Skipping sequence of length {seq_len} > max_seq_len {self.max_seq_len}"
                    )
            except StopIteration:
                self.exhausted = True

    def _get_oldest_sequence(self):
        """Return the oldest sequence across all bins."""
        if self.total_buffered == 0:
            return None
        for _, _, bin_deque in self.bins:
            if bin_deque:
                seq = bin_deque.popleft()
                self.total_buffered -= 1
                return seq
        return None

    def _get_largest_that_fits(self, remaining):
        """Get the oldest sequence from the largest bin that fits."""
        for i in range(len(self.bins) - 1, -1, -1):
            _, upper, bin_deque = self.bins[i]
            if upper <= remaining and bin_deque:
                seq = bin_deque.popleft()
                self.total_buffered -= 1
                return seq
        return None

    def _get_smallest_that_fits(self, remaining):
        """Get the oldest sequence from the smallest bin that fits."""
        for _, _, bin_deque in self.bins:
            if bin_deque and len(bin_deque[0]["tokens"]) <= remaining:
                seq = bin_deque.popleft()
                self.total_buffered -= 1
                return seq
        return None

    def __next__(self):
        if self.total_buffered == 0 and self.exhausted:
            raise StopIteration

        pack = []
        current_length = 0

        # Start with the middle bin's oldest sequence
        if self.total_buffered > 0:
            seq = self._get_oldest_sequence()
            if seq:
                pack.append(seq)
                current_length = len(seq["tokens"])
                self._fill_buffer()

        # Alternate largest and smallest
        pick_largest = True
        while current_length < self.max_seq_len:
            remaining = self.max_seq_len - current_length
            seq = (
                self._get_largest_that_fits(remaining)
                if pick_largest
                else self._get_smallest_that_fits(remaining)
            )
            if not seq:
                break
            pack.append(seq)
            current_length += len(seq["tokens"])
            pick_largest = not pick_largest
            self._fill_buffer()

        if not pack:
            raise StopIteration

        # Process pack
        tokens = [t for seq in pack for t in seq["tokens"]]
        padding = self.max_seq_len - len(tokens)
        if padding > 0:
            tokens.extend([self.padding_idx] * padding)

        # Update stats
        self.pack_count += 1
        self.total_padding_count += padding
        self.total_tokens_count += len(tokens)
        self.ages.append(1)
        # for seq in pack:
        #     age = self.current_pack_number - seq["added_pack"]
        #     self.ages.append(age)
        self.current_pack_number += 1

        return {"tokens": torch.tensor(tokens)}

    def __iter__(self):
        return self

    def _compute_stats(self):
        """Compute and return packing statistics."""
        total_time = time.time() - self.start_time
        stats = {
            "Total packs generated": self.pack_count,
            "Total processing time (s)": round(total_time, 2),
            "Average time per pack (s)": (
                round(total_time / self.pack_count, 4) if self.pack_count > 0 else 0
            ),
            "Overall padding percentage": (
                round((self.total_padding_count / self.total_tokens_count) * 100, 2)
                if self.total_tokens_count > 0
                else 0
            ),
            "Maximum age of used sequences (packs)": max(self.ages) if self.ages else 0,
            "Average age of used sequences (packs)": (
                round(sum(self.ages) / len(self.ages), 2) if self.ages else 0
            ),
        }
        return stats


if __name__ == "__main__":
    num_sequences = 500
    max_seq_len = 2048
    padding_idx = -1
    buffer_sizes = [20, 100, 500, 1000]
    min_len = 1
    max_len = max_seq_len // 2

    def generate_random_range_sequences(num, min_len, max_len):
        for _ in range(num):
            length = random.randint(min_len, max_len)
            yield {"tokens": list(range(length))}

    for buffer_size in buffer_sizes:
        print(f"\nStress Testing with buffer_size={buffer_size}")
        dataset = generate_random_range_sequences(num_sequences, min_len, max_len)
        packed_dataset = OnTheFlyPackedDataset(
            dataset, max_seq_len, padding_idx, buffer_size
        )
        iterator = iter(packed_dataset)
        for _ in iterator:
            pass
        stats = iterator._compute_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("-" * 50)
