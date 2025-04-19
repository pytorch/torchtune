# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import random

import torch
from sortedcontainers import SortedList
from torch.utils.data import IterableDataset


class OnTheFlyPackedDataset(IterableDataset):
    def __init__(
        self, dataset, max_seq_len: int, padding_idx: int = 0, buffer_size: int = 10
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
        # Buffer stores sequences, sorted by length
        self.buffer = SortedList(key=lambda x: len(x["tokens"]))
        self.exhausted = False
        self._fill_buffer()
        self.print = False

    def _fill_buffer(self):
        """Fill the buffer with sequences up to buffer_size, if available."""
        while len(self.buffer) < self.buffer_size and not self.exhausted:
            try:
                seq = next(self.iterator)
                seq_len = len(seq["tokens"])
                if seq_len <= self.max_seq_len:
                    self.buffer.add(seq)
                else:
                    if self.print:
                        print(
                            f"Skipping sequence of length {seq_len} > max_seq_len {self.max_seq_len}"
                        )
            except StopIteration:
                self.exhausted = True

    def _get_largest_that_fits(self, remaining):
        """Get the largest sequence whose length is <= remaining."""
        # Find the insertion point where all sequences to the left have length <= remaining
        idx = bisect.bisect_right(
            self.buffer, remaining, key=lambda seq: len(seq["tokens"])
        )
        if idx > 0:
            # The largest sequence that fits is just before the insertion point
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

        # Start with a random sequence
        if self.buffer:
            if self.print:
                print("buffer", [x["tokens"] for x in self.buffer])
            idx = random.randint(0, len(self.buffer) - 1)
            seq = self.buffer.pop(idx)
            pack.append(seq)
            current_length = len(seq["tokens"])
            if self.print:
                print("pack", pack)
            self._fill_buffer()

        # Alternate between largest and smallest sequences
        pick_largest = True
        while current_length < self.max_seq_len and self.buffer:
            if self.print:
                print("buffer", [x["tokens"] for x in self.buffer])
            remaining = self.max_seq_len - current_length
            if len(self.buffer[0]["tokens"]) > remaining:
                break  # Early stop if even the smallest doesn't fit
            if pick_largest:
                seq = self._get_largest_that_fits(remaining)
            else:
                seq = self._get_smallest_that_fits(remaining)
            if not seq:
                break
            pack.append(seq)
            current_length += len(seq["tokens"])
            pick_largest = not pick_largest
            if self.print:
                print("pack", pack)
            self._fill_buffer()

        if not pack:
            raise StopIteration

        # Flatten tokens and pad
        tokens = [t for seq in pack for t in seq["tokens"]]
        padding = self.max_seq_len - len(tokens)
        if padding > 0:
            tokens.extend([self.padding_idx] * padding)

        # Debugging output
        seq_lengths = [len(seq["tokens"]) for seq in pack]
        if self.print:
            print(
                f"Pack lengths: {seq_lengths}, Total: {sum(seq_lengths)}, Padded to: {len(tokens)}"
            )

        return {"tokens": torch.tensor(tokens)}

    def __iter__(self):
        return self


# 2. Function to generate range() vectors of random sequences
def generate_random_range_sequences(num_sequences, min_len=1, max_len=10):
    """Generate sequences where tokens are range(0, length)."""
    for _ in range(num_sequences):
        seq_len = random.randint(min_len, max_len)
        tokens = torch.arange(seq_len)
        yield {"tokens": tokens}


if __name__ == "__main__":
    import time

    # Stress test parameters
    num_sequences = 500  # Much larger number of sequences
    max_seq_len = 2048  # Larger sequence length for more demanding NLP tasks
    padding_idx = -1
    buffer_sizes = [20, 100, 500, 1000]  # Wider range of buffer sizes to test
    min_len = 1
    max_len = max_seq_len  # Allow sequences up to max_seq_len for variety

    for buffer_size in buffer_sizes:
        print(f"\nStress Testing with buffer_size={buffer_size}")

        # Generate a large random dataset
        dataset = generate_random_range_sequences(num_sequences, min_len, max_len)
        packed_dataset = OnTheFlyPackedDataset(
            dataset, max_seq_len, padding_idx, buffer_size
        )

        # Time the packing process
        start_time = time.time()
        pack_count = 0
        for pack in packed_dataset:
            pack_count += 1
        end_time = time.time()

        # Calculate and display results
        total_time = end_time - start_time
        avg_time_per_pack = total_time / pack_count if pack_count > 0 else 0
        print(f"Total packs generated: {pack_count}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per pack: {avg_time_per_pack:.4f} seconds")
        print("-" * 50)

# 3 & 4. Main execution with debugging output
# if __name__ == "__main__":
#     # Parameters for easy debugging
#     num_sequences = 15
#     max_seq_len = 10
#     buffer_size = 4  # Small look-ahead to see refilling in action
#     padding_idx = -1  # Distinct padding value for clarity

#     # Generate dataset
#     dataset = generate_random_range_sequences(num_sequences, min_len=1, max_len=8)
#     packed_dataset = OnTheFlyPackedDataset(
#         dataset, max_seq_len, padding_idx, buffer_size
#     )

#     if self.print:
#         print(
#             f"\nPacking {num_sequences} sequences with max_seq_len={max_seq_len}, buffer_size={buffer_size}"
#         )
#     if self.print:
#         print("Results:")
#     for i, pack in enumerate(packed_dataset):
#         if self.print:
#             print(f"Pack {i + 1}: {pack['tokens'].tolist()}")
#         if self.print:
#             print("-" * 20)
