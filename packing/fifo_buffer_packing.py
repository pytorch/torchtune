# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import time
from collections import defaultdict

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
        self.buffer = []  # Use list instead of deque
        self.exhausted = False
        self.current_pack_number = 0
        self.ages = []
        self.sequence_id_counter = 0
        self.pack_count = 0
        self.total_padding_count = 0
        self.total_tokens_count = 0
        self.start_time = time.time()
        self.length_counts = defaultdict(int)  # Dictionary for seq_len: count
        self.smallest_length = float("inf")  # Track smallest sequence length
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
                    self.buffer.append(seq)
                    self.length_counts[seq_len] += 1
                    if seq_len < self.smallest_length:
                        self.smallest_length = seq_len
                else:
                    print(
                        f"Skipping sequence of length {seq_len} > max_seq_len {self.max_seq_len}"
                    )
            except StopIteration:
                self.exhausted = True

    def _pop_and_fill(self, index):
        """Pop a sequence from the buffer at index and refill."""
        seq = self.buffer.pop(index)  # Direct pop from list
        seq_len = len(seq["tokens"])
        self.length_counts[seq_len] -= 1
        if self.length_counts[seq_len] == 0:
            del self.length_counts[seq_len]
            # Update smallest_length if it was removed
            if seq_len == self.smallest_length:
                if self.length_counts:
                    self.smallest_length = min(self.length_counts.keys())
                else:
                    self.smallest_length = float("inf")
        self._fill_buffer()
        return seq

    def __next__(self):
        if self.exhausted and not self.buffer:
            raise StopIteration

        pack = []
        current_length = 0
        buffer_idx = 0

        # Build the pack
        while current_length < self.max_seq_len and self.smallest_length <= (
            self.max_seq_len - current_length
        ):
            if buffer_idx >= len(self.buffer):
                break
            seq = self.buffer[buffer_idx]
            seq_len = len(seq["tokens"])
            if seq_len <= (self.max_seq_len - current_length):
                seq = self._pop_and_fill(buffer_idx)
                pack.append(seq)
                current_length += seq_len
                # No need to increment buffer_idx since we removed an element
            else:
                buffer_idx += 1

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
    num_sequences = 10
    max_seq_len = 10
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
        for pack in iterator:
            print(pack)
            pass
        stats = iterator._compute_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("-" * 50)
