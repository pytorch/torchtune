# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Union

import torch
from torch.nn import functional as F

from torch.utils.data import Dataset
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.utils import get_world_size_and_rank
from tqdm import tqdm

PACK_TYPE = Dict[str, Union[torch.Tensor, List[int]]]


class PackedDataset(Dataset):
    """
    Performs greedy sample packing on a provided dataset. This is done as a single
    preprocessing step before training begins. Shuffling is done outside of this
    class on packed samples with a ``Sampler`` as part of the dataloader. Currently,
    this only supports in-memory map-style datasets.

    The class loads, tokenizes, and packs examples on initialization - no tokenization is done during training.

    The general flow on initialization is: load tokenized sample -> add to buffer ->
    when buffer is long enough, add to ``self.packs``.

    During training, returns self.packs[idx] as input, label, attention mask, and
    position ids. The attention mask is a lower triangular block mask to prevent
    samples from cross-attending within a pack. The position ids indicate the position
    of each token relative to its sample within a pack. These are all padded to max
    sequence length, so a batch-wise collator is not needed.

    A packed sample is made up of individual smaller sequence length samples jammed together
    within ``max_seq_len``. For example, if max_seq_len is 6 and there are varied
    length samples::

        tokens = [
            [S1, S1, S1, S2, S2, pad],
            [S3, S3, S4, S4, pad, pad],
            ...,
        ]

    To prevent cross-contamination, the following mask would be returned for the
    first pack in the example::

        mask = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]

    The position ids would be::

        input_pos = [
            [0, 1, 2, 0, 1, 2],
            [0, 1, 0, 1, 2, 3],
            ...,
        ]

    The identity matrix is used in the mask for pad tokens instead of a causal mask.
    For position ids for pad tokens, we simply continue to increment from the previous
    sample normally.

    Args:
        ds (Dataset): dataset to sample pack. This should return a dictionary with field
            "tokens" and "labels" containing the tokenized and label samples.
        max_seq_len (int): Maximum number of tokens to pack
        padding_idx (int): padding index for the tokenizer. Default is 0.
        max_packs (Optional[int]): Maximum number of packs. Default is None, which will create as many
            packs as possible.
        split_across_pack (bool): if the last sample in a pack does not fit in ``max_seq_len``,
            split the sample into the next pack, or move it entirely to the beginning of the next pack.
            For pre-training, typically this is set to True for general text completion. For
            fine-tuning, typically this is set to False to avoid truncating sentences in instruct
            tuning. Default is False.
    """

    def __init__(
        self,
        ds: Dataset,
        *,
        max_seq_len: int,
        padding_idx: int = 0,
        max_packs: Optional[int] = None,
        split_across_pack: bool = False,
    ) -> None:
        self.ds = ds
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.max_packs = max_packs
        self.split_across_pack = split_across_pack
        # Where final samples will be held
        self.packs: List[PACK_TYPE] = []
        self.previous_sample_boundary: int = 0
        self._pack()

    def _pack(self) -> None:
        """Iterate through the dataset. Use a buffer to hold samples until max_seq_len,
        then append the buffer to self.packs as a single "packed" sample. Continue
        until max_packs or end of dataset."""
        # Buffer to hold samples until they are long enough to be added to self.packs
        current_pack = {
            "tokens": [],
            "labels": [],
            "input_pos": [],
            "seq_lens": [],
        }

        # Only show progress bar on rank 0
        _, rank = get_world_size_and_rank()
        if rank == 0:
            pbar = tqdm(total=len(self.ds), desc="Packing dataset", dynamic_ncols=True)

        for sample in self.ds:
            tokens, labels = sample["tokens"], sample["labels"]

            # If the dataset outputs samples that are larger than the specified
            # max_seq_len and we're unable to split it, user needs to modify
            # one of the two parameters
            seq_len = len(tokens)
            if seq_len > self.max_seq_len and not self.split_across_pack:
                raise ValueError(
                    f"Dataset sample is too long ({seq_len} > {self.max_seq_len}). "
                    "Please set `split_across_pack=True` or increase `max_seq_len`."
                )

            # Update the current pack
            current_pack["tokens"] += tokens
            current_pack["labels"] += labels
            current_pack["input_pos"] += list(range(seq_len))
            current_pack["seq_lens"] += [seq_len]

            # If the current pack is over the max_seq_len, add it to self.packs and
            # retain any truncated or bumped samples for next pack
            if len(current_pack["tokens"]) > self.max_seq_len:
                current_pack = self._split_and_add_pack(current_pack)

            if rank == 0:
                pbar.update()

            # Keep track of previous sample boundary
            self.previous_sample_boundary = len(current_pack["tokens"])

            # If max packs is set, stop packing when we reach that number
            if self.max_packs is not None and len(self.packs) == self.max_packs:
                break

        # Handle the last pack if there's leftover and we haven't filled up the max packs
        if len(current_pack["tokens"]) > 0 and (
            self.max_packs is None or len(self.packs) < self.max_packs
        ):
            # No need to handle splitting at this point so we can just add the current pack
            self._add_pack(current_pack)

    def _split_and_add_pack(self, current_pack: PACK_TYPE) -> PACK_TYPE:
        """Splits the current pack at the boundary, processes it, adds it to ``self.packs`` and
        returns the start of the next pack."""

        if self.split_across_pack:
            boundary = self.max_seq_len
            # The last elem in ``seq_lens`` ensures that ``sum(seq_lens) == self.max_seq_len``
            seq_len_padding = [self.max_seq_len - sum(current_pack["seq_lens"][:-1])]
        else:
            boundary = self.previous_sample_boundary
            # If we aren't splitting across packs, we leave out the last sample b/c
            # it will go into the next pack
            seq_len_padding = []

        pack = {
            "tokens": current_pack["tokens"][:boundary],
            "labels": current_pack["labels"][:boundary],
            "input_pos": current_pack["input_pos"][:boundary],
            "seq_lens": current_pack["seq_lens"][:-1] + seq_len_padding,
        }

        # Process and add the pack
        self._add_pack(pack)

        # Return the length of the first sample in next pack if we are splitting across packs,
        # otherwise return the length of the last sample in the current pack
        next_seq_len = (
            len(current_pack["tokens"][boundary:])
            if self.split_across_pack
            else current_pack["seq_lens"][-1]
        )

        return {
            "tokens": current_pack["tokens"][boundary:],
            "labels": current_pack["labels"][boundary:],
            "input_pos": current_pack["input_pos"][boundary:],
            "seq_lens": [next_seq_len],
        }

    def _add_pack(self, pack: PACK_TYPE) -> None:
        """Processes, pads and adds a pack to ``self.packs``."""
        pack = self._convert_to_tensors(pack)
        pack = self._pad_pack(pack, padding_idx=self.padding_idx)
        self.packs.append(pack)

    def _convert_to_tensors(self, pack: PACK_TYPE) -> PACK_TYPE:
        """Converts a pack into tensors. Pack comes in as a dict of lists and is converted to tensors.
        The only key that does not get converted is ``seq_lens``.
        """
        return {
            "tokens": torch.tensor(pack["tokens"]),
            "labels": torch.tensor(pack["labels"]),
            "input_pos": torch.tensor(pack["input_pos"]),
            "seq_lens": pack["seq_lens"],
        }

    def _pad_pack(self, pack: PACK_TYPE, padding_idx: int) -> PACK_TYPE:
        """Pads a pack to ``self.max_seq_len``."""
        # Pad tokens
        padded_tokens = F.pad(
            pack["tokens"],
            (0, self.max_seq_len - len(pack["tokens"])),
            value=padding_idx,
        )

        # Pad labels
        padded_labels = F.pad(
            pack["labels"],
            (0, self.max_seq_len - len(pack["labels"])),
            value=CROSS_ENTROPY_IGNORE_IDX,
        )

        # Pad input_pos continuing the sequence from last value
        # in input_pos
        # e.g. [0 1 2] -> [0 1 2 3 4 5] for self.max_seq_len = 6
        num_range = torch.arange(
            pack["input_pos"][-1] + 1,
            pack["input_pos"][-1] + self.max_seq_len - len(pack["input_pos"]) + 1,
        )
        # Clamp to max_seq_len - 1 to avoid out of bounds error
        clamped_num_range = torch.clamp(num_range, 0, self.max_seq_len - 1)
        padded_input_pos = torch.cat([pack["input_pos"], clamped_num_range])

        return {
            "tokens": padded_tokens,
            "labels": padded_labels,
            "input_pos": padded_input_pos,
            "seq_lens": pack["seq_lens"],  # seq_len is untouched
        }

    def __len__(self) -> int:
        return len(self.packs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Constructs the attention mask on-the-fly and returns whole sample."""
        current_pack = self.packs[idx]

        num_samples_in_pack = len(current_pack["seq_lens"])
        total_seq_len = 0

        block_attn_masks = []

        for i, seq_len in enumerate(current_pack["seq_lens"]):
            total_seq_len += seq_len

            # Append lower triangular matrix for causal mask
            block_attn_masks.append(
                torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            )

            # If we're at the last sample and the total seq len is less than the max seq len,
            # we need to pad with identity matrix for the remainder
            if i == num_samples_in_pack - 1 and total_seq_len < self.max_seq_len:
                block_attn_masks.append(
                    torch.eye(
                        self.max_seq_len - total_seq_len,
                        self.max_seq_len - total_seq_len,
                        dtype=torch.bool,
                    )
                )

        return {
            "tokens": current_pack["tokens"],
            "labels": current_pack["labels"],
            "input_pos": current_pack["input_pos"],
            # Assemble the mask into a block causal matrix
            "mask": torch.block_diag(*block_attn_masks),
        }
