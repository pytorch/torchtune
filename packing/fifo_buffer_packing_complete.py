# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from collections import defaultdict

import torch
from torch.utils.data import IterableDataset

CROSS_ENTROPY_IGNORE_IDX = -100


class OnTheFlyPackedDataset(IterableDataset):
    def __init__(
        self,
        dataset,
        max_seq_len: int,
        padding_idx: int = -1,
        buffer_size: int = 10,
        split_across_pack: bool = False,
    ):
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.buffer_size = buffer_size
        self.split_across_pack = split_across_pack

    def __iter__(self):
        return EfficientPackedIterator(
            self.dataset,
            self.max_seq_len,
            self.padding_idx,
            self.buffer_size,
            self.split_across_pack,
        )


class EfficientPackedIterator:
    def __init__(
        self, dataset, max_seq_len, padding_idx, buffer_size, split_across_pack
    ):
        self.iterator = iter(dataset)
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.buffer_size = buffer_size
        self.split_across_pack = split_across_pack
        self.buffer = []
        self.exhausted = False
        self.length_counts = defaultdict(int)
        self.smallest_length = float("inf")
        self.remainder_seq = None
        self._fill_buffer()

    def _fill_buffer(self):
        """Fill the buffer up to buffer_size with sequences."""
        while len(self.buffer) < self.buffer_size and not self.exhausted:
            try:
                seq = next(self.iterator)
                seq_len = len(seq["tokens"])
                if seq_len <= self.max_seq_len or self.split_across_pack:
                    self.buffer.append(seq)
                    self.length_counts[seq_len] += 1
                    self.smallest_length = min(self.smallest_length, seq_len)
                else:
                    print(
                        f"Skipping sequence of length {seq_len} > max_seq_len {self.max_seq_len}"
                    )
            except StopIteration:
                self.exhausted = True

    def _pop_and_fill(self, index):
        """Pop a sequence from the buffer at index and refill."""
        seq = self.buffer.pop(index)
        seq_len = len(seq["tokens"])

        # Update smallest_length for early stopping
        self.length_counts[seq_len] -= 1
        if self.length_counts[seq_len] == 0:
            del self.length_counts[seq_len]
            if seq_len == self.smallest_length:
                self.smallest_length = (
                    min(self.length_counts.keys())
                    if self.length_counts
                    else float("inf")
                )

        self._fill_buffer()
        return seq

    def _create_empty_pack(self):
        """Create an empty pack structure."""
        return {"tokens": [], "labels": [], "input_pos": [], "seq_lens": []}

    def _add_sequence_to_pack(self, pack, sequence, length=None):
        """Add a sequence or slice to the pack."""
        length = length or len(sequence["tokens"])
        pack["tokens"].extend(sequence["tokens"][:length])
        pack["labels"].extend(sequence["labels"][:length])
        pack["input_pos"].extend(range(length))
        pack["seq_lens"].append(length)
        return length

    def _pad_pack(self, pack):
        """Pad the pack to max_seq_len."""
        num_padding = self.max_seq_len - len(pack["tokens"])
        if num_padding > 0:
            pack["tokens"].extend([self.padding_idx] * num_padding)
            pack["labels"].extend([CROSS_ENTROPY_IGNORE_IDX] * num_padding)
            last_pos = pack["input_pos"][-1] if pack["input_pos"] else -1
            pack["input_pos"].extend(range(last_pos + 1, last_pos + 1 + num_padding))
            pack["seq_lens"][-1] += num_padding  # Add padding length to seq_lens

    # def _create_block_causal_mask(self, seq_lens):
    #     """Generate a block diagonal causal mask."""
    #     mask = torch.zeros((self.max_seq_len, self.max_seq_len), dtype=torch.bool)
    #     start = 0
    #     for length in seq_lens:
    #         end = start + length
    #         mask[start:end, start:end] = torch.tril(
    #             torch.ones((length, length), dtype=torch.bool)
    #         )
    #         start = end
    #     return mask

    def _finalize_pack(self, pack):
        """Pad the pack and convert lists to tensors."""
        self._pad_pack(pack)
        pack["tokens"] = torch.tensor(pack["tokens"], dtype=torch.long)
        pack["labels"] = torch.tensor(pack["labels"], dtype=torch.long)
        pack["input_pos"] = torch.tensor(pack["input_pos"], dtype=torch.long)
        pack["seq_lens"] = torch.tensor(pack["seq_lens"], dtype=torch.long)
        # pack["mask"] = self._create_block_causal_mask(pack["seq_lens"])
        return pack

    def _next_without_splitting(self):
        if self.exhausted and not self.buffer:
            raise StopIteration

        pack = self._create_empty_pack()
        current_length = 0
        buffer_idx = 0

        while (
            buffer_idx < len(self.buffer)
            and current_length < self.max_seq_len
            and self.smallest_length <= (self.max_seq_len - current_length)
        ):

            seq = self.buffer[buffer_idx]
            seq_len = len(seq["tokens"])

            if seq_len <= (self.max_seq_len - current_length):
                seq = self._pop_and_fill(buffer_idx)  # Pop, buffer shifts, idx stays
                current_length += self._add_sequence_to_pack(pack, seq)
            else:
                buffer_idx += 1  # Check next if it doesn't fit

        if pack["tokens"]:
            return self._finalize_pack(pack)
        else:
            raise StopIteration

    def _next_with_splitting(self):
        if self.exhausted and not self.buffer and not self.remainder_seq:
            raise StopIteration

        pack = self._create_empty_pack()
        current_length = 0

        # Handle remainder from the previous pack
        if self.remainder_seq:
            seq = self.remainder_seq
            seq_len = len(seq["tokens"])
            fit_len = min(seq_len, self.max_seq_len - current_length)

            current_length += self._add_sequence_to_pack(pack, seq, fit_len)

            # remainder is longer than the pack, so we need to split it again
            if fit_len < seq_len:
                self.remainder_seq = {
                    "tokens": seq["tokens"][fit_len:],
                    "labels": seq["labels"][fit_len:],
                }
                return self._finalize_pack(pack)

            self.remainder_seq = None

        # Fill the pack with sequences from the buffer
        buffer_idx = 0
        while current_length < self.max_seq_len and buffer_idx < len(self.buffer):
            seq = self.buffer[buffer_idx]
            seq_len = len(seq["tokens"])

            if seq_len <= (self.max_seq_len - current_length):
                seq = self._pop_and_fill(buffer_idx)
                current_length += self._add_sequence_to_pack(pack, seq)
            else:
                # Split the sequence
                fit_len = self.max_seq_len - current_length
                current_length += self._add_sequence_to_pack(pack, seq, fit_len)

                self.remainder_seq = {
                    "tokens": seq["tokens"][fit_len:],
                    "labels": seq["labels"][fit_len:],
                }
                self.buffer.pop(buffer_idx)
                self._fill_buffer()
                break

        if pack["tokens"]:
            return self._finalize_pack(pack)
        else:
            raise StopIteration

    def __next__(self):
        if self.split_across_pack:
            return self._next_with_splitting()
        else:
            return self._next_without_splitting()

    def __iter__(self):
        return self


# Example usage
if __name__ == "__main__":

    def generate_random_sequences(num, min_len, max_len):
        for _ in range(num):
            length = random.randint(min_len, max_len)
            yield {"tokens": list(range(length)), "labels": list(range(length))}

    dataset = generate_random_sequences(30, 1, 10)
    packed_dataset = OnTheFlyPackedDataset(
        dataset, max_seq_len=10, buffer_size=30, split_across_pack=False
    )
    iterator = iter(packed_dataset)
    for pack in iterator:
        print(f"Tokens: {pack['tokens'].tolist()}")
        print(f"input_pos:\n {pack['input_pos']}")
        print(f"Seq Lens: {pack['seq_lens'].tolist()}")
        print("-" * 50)
