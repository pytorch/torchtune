# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

        # Initialize bins: each bin is (lower, upper, deque)
        self.bins = []
        self.bin_size = max_seq_len // num_bins
        for i in range(num_bins):
            lower = int(i * self.bin_size)
            upper = int((i + 1) * self.bin_size) if i < num_bins - 1 else max_seq_len
            self.bins.append((lower, upper, deque()))

        self.total_buffered = 0  # Total sequences across all bins
        self.exhausted = False
        self.current_pack_number = 0  # Number of packs yielded
        self.ages = []  # List to store ages of used sequences

        # Fill the buffer initially
        self._fill_buffer()

    def _fill_buffer(self):
        """Fill bins with sequences until total_buffered reaches buffer_size or dataset is exhausted."""
        while self.total_buffered < self.buffer_size and not self.exhausted:
            try:
                seq = next(self.iterator)
                seq_len = len(seq["tokens"])
                if seq_len <= self.max_seq_len:
                    # Assign pack number when added
                    seq["added_pack"] = self.current_pack_number
                    # Find bin: (seq_len - 1) // bin_size gives bin index
                    bin_index = min(
                        max(0, (seq_len - 1) // self.bin_size), self.num_bins - 1
                    )
                    self.bins[bin_index][2].append(seq)
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
        # Find bin with the oldest sequence (smallest added_pack)
        candidates = [
            (bin_deque[0]["added_pack"], bin_idx)
            for bin_idx, (_, _, bin_deque) in enumerate(self.bins)
            if bin_deque
        ]
        if not candidates:
            return None
        _, bin_idx = min(candidates)
        seq = self.bins[bin_idx][2].popleft()
        self.total_buffered -= 1
        return seq

    def _get_largest_that_fits(self, remaining):
        """Get the oldest sequence from the largest bin where upper <= remaining."""
        for bin_idx in reversed(range(len(self.bins))):
            _, upper, bin_deque = self.bins[bin_idx]
            if upper <= remaining and bin_deque:
                seq = bin_deque.popleft()
                self.total_buffered -= 1
                return seq
        return None

    def _get_smallest_that_fits(self, remaining):
        """Get the oldest sequence from the smallest bin where sequence fits."""
        for _, upper, bin_deque in self.bins:
            if bin_deque and len(bin_deque[0]["tokens"]) <= remaining:
                seq = bin_deque.popleft()
                self.total_buffered -= 1
                return seq
        return None

    def __next__(self):
        """Yield the next packed sequence."""
        if self.total_buffered == 0 and self.exhausted:
            raise StopIteration

        pack = []
        current_length = 0

        # Start with the oldest sequence
        seq = self._get_oldest_sequence()
        if seq:
            pack.append(seq)
            current_length = len(seq["tokens"])
            self._fill_buffer()

        # Alternate between largest and smallest bins
        pick_largest = True
        while current_length < self.max_seq_len:
            remaining = self.max_seq_len - current_length
            seq = self._get_largest_that_fits(remaining)
            # seq = (
            #     self._get_largest_that_fits(remaining)
            #     if pick_largest
            #     else self._get_smallest_that_fits(remaining)
            # )
            if not seq:
                break
            pack.append(seq)
            current_length += len(seq["tokens"])
            pick_largest = not pick_largest
            self._fill_buffer()

        if not pack:
            raise StopIteration

        # Flatten tokens and pad
        tokens = [t for seq in pack for t in seq["tokens"]]
        padding = self.max_seq_len - len(tokens)
        if padding > 0:
            tokens.extend([self.padding_idx] * padding)

        # Calculate ages of sequences in the pack
        c = 0
        for _, upper, bin_deque in self.bins:
            # print(f"Bin {c} has {len(bin_deque)} sequences")
            # c += 1
            if bin_deque:
                seq = bin_deque[0]
                age = self.current_pack_number - seq["added_pack"]
                self.ages.append(age)

        # Increment pack number
        self.current_pack_number += 1

        return {"tokens": torch.tensor(tokens)}

    def __iter__(self):
        return self

    def print_statistics(self):
        """Print statistics about sequence ages."""
        if self.ages:
            max_age = max(self.ages)
            avg_age = sum(self.ages) / len(self.ages)
            print(f"Maximum age of used sequences: {max_age} packs")
            print(f"Average age of used sequences: {avg_age:.2f} packs")
        else:
            print("No sequences were processed.")


# Example usage remains the same as in your original code
if __name__ == "__main__":
    import random
    import time

    def generate_random_range_sequences(num_sequences, min_len=1, max_len=10):
        """Generate sequences where tokens are range(0, length)."""
        for _ in range(num_sequences):
            seq_len = random.randint(min_len, max_len)
            tokens = torch.arange(seq_len)
            yield {"tokens": tokens}

    # Stress test parameters
    num_sequences = 500
    max_seq_len = 2048
    padding_idx = -1
    buffer_sizes = [20, 100, 500, 1000]
    min_len = 1
    max_len = max_seq_len // 2

    for buffer_size in buffer_sizes:
        print(f"\nStress Testing with buffer_size={buffer_size}")

        dataset = generate_random_range_sequences(num_sequences, min_len, max_len)
        packed_dataset = OnTheFlyPackedDataset(
            dataset, max_seq_len, padding_idx, buffer_size
        )
        iterator = iter(packed_dataset)

        start_time = time.time()
        pack_count = 0
        total_padding_count = 0
        total_tokens_count = 0
        for pack in iterator:
            pack_count += 1
            tokens = pack["tokens"].tolist()
            padding_count = tokens.count(padding_idx)
            total_padding_count += padding_count
            total_tokens_count += len(tokens)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_pack = total_time / pack_count if pack_count > 0 else 0
        padding_percentage = (
            (total_padding_count / total_tokens_count) * 100
            if total_tokens_count > 0
            else 0
        )

        print(f"Total packs generated: {pack_count}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per pack: {avg_time_per_pack:.4f} seconds")
        print(f"Overall padding percentage: {padding_percentage:.2f}%")
        iterator.print_statistics()
        print("-" * 50)
