# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import random
import time

import torch
from torch.utils.data import IterableDataset


class OnTheFlyPackedDataset(IterableDataset):
    def __init__(
        self, dataset, max_seq_len: int, padding_idx: int = -1, buffer_size: int = 10
    ):
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.buffer_size = buffer_size

    def __iter__(self):
        return EfficientPackedIterator(
            self.dataset, self.max_seq_len, self.padding_idx, self.buffer_size
        )


class EfficientPackedIterator:
    def __init__(self, dataset, max_seq_len, padding_idx, buffer_size):
        self.iterator = iter(dataset)
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.buffer_size = buffer_size
        self.buffer = []  # Sorted list by sequence length
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
        """Fill the buffer up to buffer_size with sequences."""
        while len(self.buffer) < self.buffer_size and not self.exhausted:
            try:
                seq = next(self.iterator)
                seq_len = len(seq["tokens"])
                if seq_len <= self.max_seq_len:
                    seq["id"] = self.sequence_id_counter
                    seq["added_pack"] = self.current_pack_number
                    self.sequence_id_counter += 1
                    bisect.insort(self.buffer, seq, key=lambda x: len(x["tokens"]))
                else:
                    print(
                        f"Skipping sequence of length {seq_len} > max_seq_len {self.max_seq_len}"
                    )
            except StopIteration:
                self.exhausted = True

    def _get_largest_that_fits(self, remaining):
        """Get the largest sequence whose length is <= remaining."""
        idx = bisect.bisect_right(
            self.buffer, remaining, key=lambda seq: len(seq["tokens"])
        )
        if idx > 0:
            return self.buffer.pop(idx - 1)
        return None

    def _get_smallest_that_fits(self, remaining):
        """Get the smallest sequence whose length is <= remaining."""
        if self.buffer and len(self.buffer[0]["tokens"]) <= remaining:
            return self.buffer.pop(0)
        return None

    def __next__(self):
        if self.exhausted and not self.buffer:
            raise StopIteration

        pack = []
        current_length = 0

        # Start with the middle sequence
        if self.buffer:
            idx = len(self.buffer) // 2
            seq = self.buffer.pop(idx)
            pack.append(seq)
            current_length = len(seq["tokens"])
            self._fill_buffer()

        # Alternate largest and smallest
        pick_largest = True
        while current_length < self.max_seq_len and self.buffer:
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
