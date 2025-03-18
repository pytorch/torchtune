# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch.nn import functional as F

from torch.utils.data import Dataset, IterableDataset
from torchdata.nodes import Stateful
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX, PACK_TYPE
from tqdm import tqdm


# class PackedDataset(Dataset):
#     """
#     Performs greedy sample packing on a provided dataset. This is done as a single
#     preprocessing step before training begins. Shuffling is done outside of this
#     class on packed samples with a ``Sampler`` as part of the dataloader. Currently,
#     this only supports in-memory map-style datasets.

#     The class loads, tokenizes, and packs examples on initialization - no tokenization is done during training.

#     The general flow on initialization is: load tokenized sample -> add to buffer ->
#     when buffer is long enough, add to ``self.packs``.

#     During training, returns self.packs[idx] as input, label, attention mask, and
#     position ids. The attention mask is a lower triangular block mask to prevent
#     samples from cross-attending within a pack. The position ids indicate the position
#     of each token relative to its sample within a pack. These are all padded to max
#     sequence length, so a batch-wise collator is not needed.

#     A packed sample is made up of individual smaller sequence length samples jammed together
#     within ``max_seq_len``. For example, if max_seq_len is 6 and there are varied
#     length samples::

#         tokens = [
#             [S1, S1, S1, S2, S2, pad],
#             [S3, S3, S4, S4, pad, pad],
#             ...,
#         ]

#     To prevent cross-contamination, the following mask would be returned for the
#     first pack in the example::

#         mask = [
#             [1, 0, 0, 0, 0, 0],
#             [1, 1, 0, 0, 0, 0],
#             [1, 1, 1, 0, 0, 0],
#             [0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 1, 1, 0],
#             [0, 0, 0, 0, 0, 1],
#         ]

#     The position ids would be::

#         input_pos = [
#             [0, 1, 2, 0, 1, 2],
#             [0, 1, 0, 1, 2, 3],
#             ...,
#         ]

#     The identity matrix is used in the mask for pad tokens instead of a causal mask.
#     For position ids for pad tokens, we simply continue to increment from the previous
#     sample normally.

#     Args:
#         ds (Dataset): dataset to sample pack. This should return a dictionary with field
#             "tokens" and "labels" containing the tokenized and label samples.
#         max_seq_len (int): Maximum number of tokens to pack
#         padding_idx (int): padding index for the tokenizer. Default is 0.
#         max_packs (Optional[int]): Maximum number of packs. Default is None, which will create as many
#             packs as possible.
#         split_across_pack (bool): if the last sample in a pack does not fit in ``max_seq_len``,
#             split the sample into the next pack, or move it entirely to the beginning of the next pack.
#             For pre-training, typically this is set to True for general text completion. For
#             fine-tuning, typically this is set to False to avoid truncating sentences in instruct
#             tuning. Default is False.
#     """

#     def __init__(
#         self,
#         ds: Dataset,
#         *,
#         max_seq_len: int,
#         padding_idx: int = 0,
#         max_packs: Optional[int] = None,
#         split_across_pack: bool = False,
#     ) -> None:
#         self.ds = ds
#         self.max_seq_len = max_seq_len
#         self.padding_idx = padding_idx
#         self.max_packs = max_packs
#         self.split_across_pack = split_across_pack
#         # Where final samples will be held
#         self.packs: List[PACK_TYPE] = []
#         self.previous_sample_boundary: int = 0
#         self._pack()

#     def _pack(self) -> None:
#         """Iterate through the dataset. Use a buffer to hold samples until max_seq_len,
#         then append the buffer to self.packs as a single "packed" sample. Continue
#         until max_packs or end of dataset."""
#         # Buffer to hold samples until they are long enough to be added to self.packs
#         current_pack = {
#             "tokens": [],
#             "labels": [],
#             "input_pos": [],
#             "seq_lens": [],
#         }

#         # Only show progress bar on rank 0
#         rank = (
#             torch.distributed.get_rank()
#             if torch.distributed.is_available() and torch.distributed.is_initialized()
#             else 0
#         )
#         if rank == 0:
#             pbar = tqdm(total=len(self.ds), desc="Packing dataset", dynamic_ncols=True)

#         for sample in self.ds:
#             tokens, labels = sample["tokens"], sample["labels"]

#             # If the dataset outputs samples that are larger than the specified
#             # max_seq_len and we're unable to split it, user needs to modify
#             # one of the two parameters
#             seq_len = len(tokens)
#             if seq_len > self.max_seq_len and not self.split_across_pack:
#                 raise ValueError(
#                     f"Dataset sample is too long ({seq_len} > {self.max_seq_len}). "
#                     "Please set `split_across_pack=True` or increase `max_seq_len`."
#                 )

#             # Update the current pack
#             current_pack["tokens"] += tokens
#             current_pack["labels"] += labels
#             current_pack["input_pos"] += [x % self.max_seq_len for x in range(seq_len)]
#             current_pack["seq_lens"] += [seq_len]

#             # If the current pack is over the max_seq_len, add it to self.packs and
#             # retain any truncated or bumped samples for next pack
#             while (
#                 len(current_pack["tokens"]) > self.max_seq_len
#                 and not self._should_stop_packing()
#             ):
#                 current_pack = self._split_and_add_pack(current_pack)

#             if rank == 0:
#                 pbar.update()

#             # Keep track of previous sample boundary
#             self.previous_sample_boundary = len(current_pack["tokens"])

#             if self._should_stop_packing():
#                 break

#         # Handle the last pack if there's leftover and we haven't filled up the max packs
#         if len(current_pack["tokens"]) > 0 and (
#             self.max_packs is None or len(self.packs) < self.max_packs
#         ):
#             # No need to handle splitting at this point so we can just add the current pack
#             self._add_pack(current_pack)

#     def _should_stop_packing(self) -> bool:
#         """If max packs is set, stop packing when we reach that number."""

#         if self.max_packs is not None and len(self.packs) == self.max_packs:
#             return True
#         return False

#     def _split_and_add_pack(self, current_pack: PACK_TYPE) -> PACK_TYPE:
#         """Splits the current pack at the boundary, processes it, adds it to ``self.packs`` and
#         returns the start of the next pack."""

#         if self.split_across_pack:
#             boundary = self.max_seq_len
#             # The last elem in ``seq_lens`` ensures that ``sum(seq_lens) == self.max_seq_len``
#             leftover_seq_len = self.max_seq_len - sum(current_pack["seq_lens"][:-1])
#             seq_len_padding = [leftover_seq_len] if leftover_seq_len > 0 else []
#         else:
#             boundary = self.previous_sample_boundary
#             # If we aren't splitting across packs, we leave out the last sample b/c
#             # it will go into the next pack
#             seq_len_padding = []

#         pack = {
#             "tokens": current_pack["tokens"][:boundary],
#             "labels": current_pack["labels"][:boundary],
#             "input_pos": current_pack["input_pos"][:boundary],
#             "seq_lens": current_pack["seq_lens"][:-1] + seq_len_padding,
#         }

#         # Process and add the pack
#         self._add_pack(pack)

#         # Return the length of the first sample in next pack if we are splitting across packs,
#         # otherwise return the length of the last sample in the current pack
#         next_seq_len = (
#             len(current_pack["tokens"][boundary:])
#             if self.split_across_pack
#             else current_pack["seq_lens"][-1]
#         )

#         return {
#             "tokens": current_pack["tokens"][boundary:],
#             "labels": current_pack["labels"][boundary:],
#             "input_pos": current_pack["input_pos"][boundary:],
#             "seq_lens": [next_seq_len],
#         }

#     def _add_pack(self, pack: PACK_TYPE) -> None:
#         """Processes, pads and adds a pack to ``self.packs``."""
#         pack = self._convert_to_tensors(pack)
#         pack = self._pad_pack(pack, padding_idx=self.padding_idx)
#         self.packs.append(pack)

#     def _convert_to_tensors(self, pack: PACK_TYPE) -> PACK_TYPE:
#         """Converts a pack into tensors. Pack comes in as a dict of lists and is converted to tensors."""
#         return {
#             "tokens": torch.tensor(pack["tokens"], dtype=torch.long),
#             "labels": torch.tensor(pack["labels"], dtype=torch.long),
#             "input_pos": torch.tensor(pack["input_pos"], dtype=torch.long),
#             "seq_lens": torch.tensor(pack["seq_lens"], dtype=torch.long),
#         }

#     def _pad_pack(self, pack: PACK_TYPE, padding_idx: int) -> PACK_TYPE:
#         """Pads a pack to ``self.max_seq_len``."""
#         # Pad tokens
#         num_padding_tokens = self.max_seq_len - len(pack["tokens"])
#         padded_tokens = F.pad(
#             pack["tokens"],
#             (0, num_padding_tokens),
#             value=padding_idx,
#         )

#         # Pad labels
#         padded_labels = F.pad(
#             pack["labels"],
#             (0, self.max_seq_len - len(pack["labels"])),
#             value=CROSS_ENTROPY_IGNORE_IDX,
#         )

#         # Add padding tokens as a last seq len to ensure sum is max_seq_len
#         padded_seq_lens = (
#             torch.cat([pack["seq_lens"], torch.tensor([num_padding_tokens])])
#             if num_padding_tokens > 0
#             else pack["seq_lens"]
#         )

#         # Pad input_pos continuing the sequence from last value
#         # in input_pos
#         # e.g. [0 1 2] -> [0 1 2 3 4 5] for self.max_seq_len = 6
#         num_range = torch.arange(
#             pack["input_pos"][-1] + 1,
#             pack["input_pos"][-1] + self.max_seq_len - len(pack["input_pos"]) + 1,
#         )
#         # Clamp to max_seq_len - 1 to avoid out of bounds error
#         clamped_num_range = torch.clamp(num_range, 0, self.max_seq_len - 1)
#         padded_input_pos = torch.cat([pack["input_pos"], clamped_num_range])

#         return {
#             "tokens": padded_tokens,
#             "labels": padded_labels,
#             "input_pos": padded_input_pos,
#             "seq_lens": padded_seq_lens,
#         }

#     def __len__(self) -> int:
#         return len(self.packs)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         return self.packs[idx]


class PackedDataset(IterableDataset, Stateful):
    def __init__(
        self,
        ds,
        max_seq_len: int,
        padding_idx: int = 0,
        buffer_size: int = 100,
        split_across_pack: bool = False,
    ):
        self.dataset = ds
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

    def state_dict(self):
        """Save the current state of the iterator."""
        state = {
            "buffer": self.buffer,
            "exhausted": self.exhausted,
            "length_counts": dict(self.length_counts),  # Convert defaultdict to dict
            "smallest_length": self.smallest_length,
            "remainder_seq": self.remainder_seq,
        }

        # Optionally include dataset state if it’s stateful
        if hasattr(self.dataset, "state_dict"):
            state["dataset_state"] = self.dataset.state_dict()
        else:
            raise ValueError("Dataset is not stateful. Cannot save state.")
        return state

    def load_state_dict(self, state_dict):
        """Restore the iterator state from a state dictionary."""
        # Restore dataset state if provided
        if "dataset_state" in state_dict and hasattr(self.dataset, "load_state_dict"):
            self.dataset.load_state_dict(state_dict["dataset_state"])
        else:
            raise ValueError("Dataset is not stateful. Cannot load state.")

        # Restore iterator-specific state
        self.buffer = state_dict["buffer"]
        self.exhausted = state_dict["exhausted"]
        self.length_counts = defaultdict(int, state_dict["length_counts"])
        self.smallest_length = state_dict["smallest_length"]
        self.remainder_seq = state_dict["remainder_seq"]

        # Reinitialize the iterator based on the restored state
        self.iterator = iter(self.dataset)


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

    def _finalize_pack(self, pack):
        """Pad the pack and convert lists to tensors."""
        self._pad_pack(pack)
        pack["tokens"] = torch.tensor(pack["tokens"], dtype=torch.long)
        pack["labels"] = torch.tensor(pack["labels"], dtype=torch.long)
        pack["input_pos"] = torch.tensor(pack["input_pos"], dtype=torch.long)
        pack["seq_lens"] = torch.tensor(pack["seq_lens"], dtype=torch.long)

        # print(f"Pack: {pack['tokens'].shape}")
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
