# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset
from tqdm import tqdm


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
            "input_ids": [],
            "labels": [],
        }
        # Keep track of what index the previous sample ends in case we need
        # to end a pack early
        previous_sample_boundary = 0

        for input_ids, labels in tqdm(
            self.ds, desc="Packing dataset", dynamic_ncols=True
        ):
            # If the dataset outputs samples that are larger than the specified
            # max_seq_len and we're unable to split it, user needs to modify
            # one of the two parameters
            if len(input_ids) > self.max_seq_len and not self.split_samples:
                raise ValueError(
                    f"Dataset sample is too long ({len(input_ids)} > {self.max_seq_len}). "
                    "Please set `split_samples=True` or increase `max_seq_len`."
                )

            current_pack["input_ids"].extend(input_ids)
            current_pack["labels"].extend(labels)

            if len(current_pack["input_ids"]) > self.max_seq_len:
                current_pack = self._add_pack(
                    current_pack=current_pack,
                    boundary=self.max_seq_len
                    if self.split_samples
                    else previous_sample_boundary,
                )

            previous_sample_boundary = len(current_pack["input_ids"])
            if self.max_rows is not None and len(self.samples) >= self.max_rows:
                break

        if len(current_pack["input_ids"]) > 0 and (
            self.max_rows is None or len(self.samples) < self.max_rows
        ):
            self.samples.append(dict(current_pack))

    def _add_pack(
        self, current_pack: Dict[str, List[int]], boundary: int
    ) -> Dict[str, List[int]]:
        """
        Add the current pack to self.samples and return what's remaining of the pack.
        """
        self.samples.append({k: v[:boundary] for k, v in current_pack.items()})
        return {k: v[boundary:] for k, v in current_pack.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self.samples[index]["input_ids"], self.samples[index]["labels"]
