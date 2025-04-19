# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from collections import deque

import torch
from torch.utils.data import IterableDataset

"""
definetely sorting is much better, but more complex.

Alternative:
1. truncate
2. Try to use bins, so its still FIFO, but mixed with sorting

Lets try 2, and then compliment with 1, to try to get to 100% utilization
"""


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
        self.buffer = deque()  # Buffer to hold sequences
        self.exhausted = False
        self.smallest_length = float("inf")  # Tracks smallest sequence length in buffer
        self._fill_buffer()  # Initial buffer fill

    def _fill_buffer(self):
        """Fill the buffer with sequences up to buffer_size."""
        while len(self.buffer) < self.buffer_size and not self.exhausted:
            try:
                seq = next(self.iterator)
                seq_len = len(seq["tokens"])
                self.buffer.append(seq)
                # Update smallest_length when adding a new sequence
                if seq_len < self.smallest_length:
                    self.smallest_length = seq_len
            except StopIteration:
                self.exhausted = True

    def _pop_and_fill(self, index):
        """
        Helper function to:
        1. Pop a sequence from the buffer at the given index.
        2. Immediately call _fill_buffer to replenish the buffer.
        3. Update the smallest_length if necessary.
        """
        # Convert deque to list to allow popping from a specific index
        buffer_list = list(self.buffer)
        seq = buffer_list.pop(index)  # Pop the element at the specified index
        # Convert back to deque
        self.buffer = deque(buffer_list)

        # Refill the buffer
        self._fill_buffer()

        # Update smallest_length based on remaining sequences
        if len(self.buffer) > 0:
            self.smallest_length = min(len(seq["tokens"]) for seq in self.buffer)
        else:
            self.smallest_length = float("inf")

        return seq

    def __next__(self):
        if self.exhausted and not self.buffer:
            raise StopIteration

        pack = []
        current_length = 0
        buffer_idx = 0  # Pointer to traverse the buffer

        # Build the pack
        while current_length < self.max_seq_len:
            # Traverse the buffer
            while buffer_idx < len(self.buffer):
                seq = self.buffer[buffer_idx]
                seq_len = len(seq["tokens"])
                # Check if the sequence should be popped (if it fits)
                if seq_len <= (self.max_seq_len - current_length):
                    # Pop, fill, and update smallest_length using helper
                    seq = self._pop_and_fill(buffer_idx)
                    pack.append(seq)
                    current_length += seq_len
                else:
                    buffer_idx += 1

            # Early stopping: if smallest sequence doesn’t fit, break
            if self.smallest_length > (self.max_seq_len - current_length):
                break

        if not pack:
            raise StopIteration

        # Flatten tokens and pad to max_seq_len
        tokens = [t for seq in pack for t in seq["tokens"]]
        padding = self.max_seq_len - len(tokens)
        if padding > 0:
            tokens.extend([self.padding_idx] * padding)

        return {"tokens": torch.tensor(tokens)}

    def __iter__(self):
        return self


# Example usage
def generate_random_sequences(num_sequences, min_len=1, max_len=10):
    """Generate simple test sequences."""
    import random

    for _ in range(num_sequences):
        seq_len = random.randint(min_len, max_len)
        tokens = torch.arange(seq_len)
        yield {"tokens": tokens}


def generate_random_range_sequences(num_sequences, min_len=1, max_len=10):
    """Generate sequences where tokens are range(0, length)."""
    for _ in range(num_sequences):
        seq_len = random.randint(min_len, max_len)
        tokens = torch.arange(seq_len)
        yield {"tokens": tokens}


if __name__ == "__main__":
    import time

    # Stress test parameters
    num_sequences = 500
    max_seq_len = 2048
    padding_idx = -1
    buffer_sizes = [20, 100, 500, 1000]
    min_len = 1
    max_len = max_seq_len // 2

    for buffer_size in buffer_sizes:
        print(f"\nStress Testing with buffer_size={buffer_size}")

        # Generate a large random dataset
        dataset = generate_random_range_sequences(num_sequences, min_len, max_len)
        packed_dataset = OnTheFlyPackedDataset(
            dataset, max_seq_len, padding_idx, buffer_size
        )
        iterator = iter(packed_dataset)

        # Time the packing process
        start_time = time.time()
        pack_count = 0
        total_padding_count = 0
        total_tokens_count = 0
        for pack in iterator:
            pack_count += 1
            # Accumulate padding and token counts
            tokens = pack["tokens"].tolist()
            padding_count = tokens.count(padding_idx)
            total_padding_count += padding_count
            total_tokens_count += len(tokens)
        end_time = time.time()

        # Calculate and display results
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

        # iterator.print_statistics()
        print("-" * 50)
