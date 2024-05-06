# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

from torch.utils.data import Dataset
from tqdm import tqdm
import torch


class PackedDataset(Dataset):
    """
    Performs greedy sample packing on a provided dataset. This is done as a single
    preprocessing step before training begins. Shuffling is done outside of this
    class on packed samples as part of the dataloader.

    The class loads, tokenizes, and packs examples on initialization - no tokenization is done during training.

    The general flow on initialization is: load tokenized sample -> add to buffer ->
        when buffer is long enough, add to self.samples.

    During training, returns self.samples[idx] as input and label.

    Args:
        ds (Dataset): dataset to sample pack. This should return a tuple of tokenized
            inputs and labels.
        max_seq_len (int): Maximum number of tokens to pack
        max_rows (Optional[int]): maximum number of samples to pack. Default is None, which will pack as many samples as possible.
        split_samples (bool): if the last sample in a pack does not fit in ``max_seq_len``,
            split the sample into the next pack, or move it to the beginning of the next pack.
            For pre-training, typically this is set to True for general text completion. For
            fine-tuning, typically this is set to False to avoid truncating sentences in instruct
            tuning. Default is False.
    """

    def __init__(
        self,
        ds: Dataset,
        max_seq_len: int,
        max_rows: Optional[int] = None,
        split_samples: bool = False,
    ) -> None:
        self.ds = ds
        self.max_seq_len = max_seq_len
        self.max_rows = max_rows
        self.split_samples = split_samples
        # where final samples will be held
        self.samples: List[Dict[str, List[int]]] = []
        self._pack()

    def _pack(self) -> None:
        """
        Iterate through the dataset. Use a buffer to hold samples until max_seq_len,
        then append the buffer to self.samples as a single "packed" sample. Continue
        until max_rows or end of dataset.
        """
        # buffer to hold samples until they are long enough to be added to self.samples
        current_pack = {
            "tokens": [],
            "labels": [],
            "mask": [],
            "input_pos": [],
        }
        # Keep track of what index the previous sample ends in case we need
        # to end a pack early
        previous_sample_boundary = 0

        for tokens, labels in tqdm(
            self.ds, desc="Packing dataset", dynamic_ncols=True
        ):
            # If the dataset outputs samples that are larger than the specified
            # max_seq_len and we're unable to split it, user needs to modify
            # one of the two parameters
            seq_len = len(tokens)
            if seq_len > self.max_seq_len and not self.split_samples:
                raise ValueError(
                    f"Dataset sample is too long ({len(tokens)} > {self.max_seq_len}). "
                    "Please set `split_samples=True` or increase `max_seq_len`."
                )

            # Create integer mask and position ids for current sample
            current_sample = {
                "tokens": tokens,
                "labels": labels,
                # Mask is simply a causal mask within this sample length
                "mask": [torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))],
                "input_pos": list(range(seq_len)),
            }
            current_pack = {k: v + current_sample[k] for k, v in current_pack.items()}

            if len(current_pack["tokens"]) > self.max_seq_len:
                current_pack = self._add_pack(
                    current_pack=current_pack,
                    boundary=self.max_seq_len
                    if self.split_samples
                    else previous_sample_boundary,
                )

            previous_sample_boundary = len(current_pack["tokens"])
            if self.max_rows is not None and len(self.samples) >= self.max_rows:
                break

        if len(current_pack["tokens"]) > 0 and (
            self.max_rows is None or len(self.samples) < self.max_rows
        ):
            current_pack = self._add_pack(current_pack=current_pack, boundary=len(current_pack["tokens"]))
            assert len(current_pack["tokens"]) == 0

    def _add_pack(
        self, current_pack: Dict[str, List[int]], boundary: int
    ) -> Dict[str, List[int]]:
        """
        Add the current pack to self.samples and return what's remaining of the pack.
        """
        packing_mask = torch.block_diag(*current_pack["mask"])
        pack = {
            "tokens": current_pack["tokens"][:boundary],
            "labels": current_pack["labels"][:boundary],
            "mask": packing_mask[:boundary, :boundary],
            "input_pos": current_pack["input_pos"][:boundary],
        }

        self.samples.append(pack)

        updated_pack = {
            "tokens": current_pack["tokens"][boundary:],
            "labels": current_pack["labels"][boundary:],
            "mask": [packing_mask[boundary:, boundary:]],
            "input_pos": current_pack["input_pos"][boundary:],
        }

        return updated_pack

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.samples[index]
