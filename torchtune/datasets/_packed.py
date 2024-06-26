# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import torch
from torch.nn import functional as F

from torch.utils.data import Dataset
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.utils import get_world_size_and_rank
from tqdm import tqdm


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
        max_packs (Optional[int]): maximum number of packs. Default is None, which will create as many
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
        max_seq_len: int,
        max_packs: Optional[int] = None,
        split_across_pack: bool = False,
    ) -> None:
        self.ds = ds
        self.max_seq_len = max_seq_len
        self.max_packs = max_packs
        self.split_across_pack = split_across_pack
        # where final samples will be held
        self.packs: List[Dict[str, List[int]]] = []
        self._pack()

    def _pack(self) -> None:
        """
        Iterate through the dataset. Use a buffer to hold samples until max_seq_len,
        then append the buffer to self.packs as a single "packed" sample. Continue
        until max_packs or end of dataset.
        """
        # buffer to hold samples until they are long enough to be added to self.packs
        current_pack = {
            "tokens": [],
            "labels": [],
            "mask": [],
            "input_pos": [],
        }
        # Keep track of what index the previous sample ends in case we need
        # to end a pack early
        previous_sample_boundary = 0

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
                    f"Dataset sample is too long ({len(tokens)} > {self.max_seq_len}). "
                    "Please set `split_across_pack=True` or increase `max_seq_len`."
                )

            # Create integer mask and position ids for current sample and extend
            # current pack
            current_sample = {
                "tokens": tokens,
                "labels": labels,
                # Mask is simply a causal mask within this sample length
                "mask": [torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))],
                "input_pos": list(range(seq_len)),
            }
            current_pack = {k: v + current_sample[k] for k, v in current_pack.items()}

            # If the current pack is long enough, add it to self.packs and retain
            # any truncated samples for next pack, if splitting samples
            if len(current_pack["tokens"]) > self.max_seq_len:
                current_pack = self._add_pack(
                    current_pack=current_pack,
                    boundary=self.max_seq_len
                    if self.split_across_pack
                    else previous_sample_boundary,
                )

            if rank == 0:
                pbar.update()
            previous_sample_boundary = len(current_pack["tokens"])
            if self.max_packs is not None and len(self.packs) >= self.max_packs:
                break

        # Add the last pack with remaining samples that did not fit in previous
        if len(current_pack["tokens"]) > 0 and (
            self.max_packs is None or len(self.packs) < self.max_packs
        ):
            current_pack = self._add_pack(
                current_pack=current_pack, boundary=len(current_pack["tokens"])
            )
            assert len(current_pack["tokens"]) == 0

    def _add_pack(
        self, current_pack: Dict[str, List[int]], boundary: int
    ) -> Dict[str, List[int]]:
        """
        Pad and add the current pack to self.packs and return what's remaining.
        """
        # So far we've kept a list of causal masks, one for each sample in the pack.
        # Now we need to combine them into a single mask for the entire pack.
        packing_mask = torch.block_diag(*current_pack["mask"])
        pack = {
            "tokens": current_pack["tokens"][:boundary],
            "labels": current_pack["labels"][:boundary],
            "mask": packing_mask[:boundary, :boundary],
            "input_pos": current_pack["input_pos"][:boundary],
        }

        # Resultant shapes after padding
        # tokens: [max_seq_len, ]
        # labels: [max_seq_len, ]
        # mask: [max_seq_len, max_seq_len]
        # input_pos: [max_seq_len, ]
        padded_pack = self._pad_pack(pack)
        self.packs.append(padded_pack)

        # Keep sample that did not fit or got truncated for the next pack
        updated_pack = {
            "tokens": current_pack["tokens"][boundary:],
            "labels": current_pack["labels"][boundary:],
            "mask": [packing_mask[boundary:, boundary:]],
            "input_pos": current_pack["input_pos"][boundary:],
        }

        return updated_pack

    def _pad_pack(
        self,
        sample: Dict[str, Any],
        padding_idx: int = 0,
        ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    ) -> Dict[str, torch.Tensor]:
        """Pad a group of packed sequences to max sequence length in the group, and
        convert integer lists to tensors. Account for attention mask and position
        ids.

        This is a sample-wise collator and should not be used with the dataloader.

        Sample should look like::
            {
                "tokens": List[int],
                "labels": List[int],
                "mask": Tensor,
                "input_pos": List[int],
            }

        Args:
            sample (Dict[str, Any]): A dictionary containing tokens, labels, mask,
                and position ids
            padding_idx (int): Padding index for input ids. Defaults to 0.
            ignore_idx (int): Padding index for labels. Defaults to -100.

        Returns:
            Collated tokens, labels, mask, and position ids.
        """
        tokens = sample["tokens"]
        labels = sample["labels"]
        mask = sample["mask"]
        input_pos = sample["input_pos"]

        # Pad to max sequence length
        tokens = F.pad(
            torch.tensor(tokens), (0, self.max_seq_len - len(tokens)), value=padding_idx
        )
        labels = F.pad(
            torch.tensor(labels), (0, self.max_seq_len - len(labels)), value=ignore_idx
        )
        # For attention mask, simply use identity matrix for the pad tokens
        mask_pad = torch.eye(self.max_seq_len - mask.shape[0], dtype=torch.bool)
        mask = torch.block_diag(mask, mask_pad)
        # For position ids, continue to increment for pad tokens
        next_pos = input_pos[-1] + 1
        input_pos_pad = torch.arange(
            next_pos, next_pos + self.max_seq_len - len(input_pos)
        )
        # Do not go beyond max_seq_len - 1
        input_pos_pad = input_pos_pad.clamp(max=self.max_seq_len - 1)
        input_pos = torch.cat(
            [
                torch.tensor(input_pos),
                input_pos_pad,
            ]
        )

        assert tokens.shape == labels.shape == input_pos.shape
        assert tokens.shape[0] == mask.shape[0]

        return {
            "tokens": tokens,
            "labels": labels,
            "mask": mask,
            "input_pos": input_pos,
        }

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.packs[index]
