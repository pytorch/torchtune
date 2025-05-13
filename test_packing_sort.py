# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import random
from collections import defaultdict

import torch
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
        # Buffer is a list, kept sorted by sequence length
        self.buffer = []
        self.exhausted = False
        # Tracking variables
        self.sequence_id_counter = 0
        self.added_pack = {}  # sequence_id -> pack number when added
        self.current_pack_number = 0
        self.ages = []  # List of ages when sequences are used
        self.print = False
        self._fill_buffer()

    def _fill_buffer(self):
        """Fill the buffer with sequences up to buffer_size, if available."""
        while len(self.buffer) < self.buffer_size and not self.exhausted:
            try:
                seq = next(self.iterator)
                seq_len = len(seq["tokens"])
                if seq_len <= self.max_seq_len:
                    # Assign a unique ID and track when added
                    seq["id"] = self.sequence_id_counter
                    self.sequence_id_counter += 1
                    self.added_pack[seq["id"]] = self.current_pack_number
                    # Insert into sorted position based on length
                    bisect.insort(self.buffer, seq, key=lambda x: len(x["tokens"]))
                else:
                    if self.print:
                        print(
                            f"Skipping sequence of length {seq_len} > max_seq_len {self.max_seq_len}"
                        )
            except StopIteration:
                self.exhausted = True
        # Log the oldest sample in the buffer
        if self.buffer:
            oldest_age = self._get_oldest_age()
            if self.print:
                print(f"Oldest sample age in buffer: {oldest_age}")

    def _get_oldest_age(self):
        """Calculate the age of the oldest sample in the buffer."""
        if not self.buffer:
            return 0
        oldest_added_pack = min(self.added_pack[seq["id"]] for seq in self.buffer)
        return self.current_pack_number - oldest_added_pack

    def _get_oldest_age_idx(self):
        """Get the index of the sequence with the oldest age in the buffer."""
        if not self.buffer:
            return None
        oldest_age = self._get_oldest_age()
        for idx, seq in enumerate(self.buffer):
            if self.added_pack[seq["id"]] == self.current_pack_number - oldest_age:
                return idx
        return None

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

        # Start with a random sequence
        if self.buffer:
            if self.print:
                print("buffer", [x["tokens"].tolist() for x in self.buffer])
            # idx = random.randint(0, len(self.buffer) - 1)
            # idx = len(self.buffer) // 2
            idx = self._get_oldest_age_idx()
            seq = self.buffer.pop(idx)
            pack.append(seq)
            current_length = len(seq["tokens"])
            if self.print:
                print("pack", [x["tokens"].tolist() for x in pack])
            self._fill_buffer()

        # Alternate between largest and smallest sequences
        pick_largest = True
        while current_length < self.max_seq_len and self.buffer:
            if self.print:
                print("buffer", [x["tokens"].tolist() for x in self.buffer])
            remaining = self.max_seq_len - current_length
            if len(self.buffer[0]["tokens"]) > remaining:
                break  # Early stop if even the smallest doesn't fit
            seq = self._get_largest_that_fits(remaining)
            # if pick_largest:
            #     seq = self._get_largest_that_fits(remaining)
            # else:
            #     seq = self._get_smallest_that_fits(remaining)
            if not seq:
                break
            pack.append(seq)
            current_length += len(seq["tokens"])
            pick_largest = not pick_largest
            if self.print:
                print("pack", [x["tokens"].tolist() for x in pack])
            self._fill_buffer()

        if not pack:
            raise StopIteration

        # Flatten tokens and pad
        tokens = [t for seq in pack for t in seq["tokens"]]
        padding = self.max_seq_len - len(tokens)
        if padding > 0:
            tokens.extend([self.padding_idx] * padding)

        # Calculate ages of sequences in the pack
        for seq in pack:
            age = self.current_pack_number - self.added_pack[seq["id"]]
            self.ages.append(age)
            # Clean up
            del self.added_pack[seq["id"]]

        # Increment pack number after completing a pack
        self.current_pack_number += 1

        # Log the oldest sample age
        oldest_age = self._get_oldest_age()
        if self.print:
            print(f"Oldest sample age in buffer after pack: {oldest_age}")

        # Debugging output
        seq_lengths = [len(seq["tokens"]) for seq in pack]
        if self.print:
            print(
                f"Pack lengths: {seq_lengths}, Total: {sum(seq_lengths)}, Padded to: {len(tokens)}"
            )

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
            # Histogram of ages
            hist = defaultdict(int)
            for age in self.ages:
                hist[age] += 1
            # print("Age histogram of used sequences:")
            # for age in sorted(hist.keys()):
            #     print(f"  Age {age}: {hist[age]} sequences")
        else:
            print("No sequences were processed.")


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
        seq_count_per_pack = []
        for pack in iterator:
            pack_count += 1
            # Accumulate padding and token counts
            count_seqs = (pack["tokens"] == 0).sum()
            seq_count_per_pack.append(count_seqs)
            tokens = pack["tokens"].tolist()
            padding_count = tokens.count(padding_idx)
            total_padding_count += padding_count
            total_tokens_count += len(tokens)
            # print("count_seqs", count_seqs, "pack", pack)

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
        print(
            f"Average number of sequences per pack: {sum(seq_count_per_pack)/len(seq_count_per_pack)}"
        )

        iterator.print_statistics()
        print("-" * 50)

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
