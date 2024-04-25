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
    """

    def __init__(
        self,
        ds: Dataset,
        max_seq_len: int,
        max_rows: Optional[int] = None,
    ) -> None:
        self.ds = ds
        self.max_seq_len = max_seq_len
        self.max_rows = max_rows
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
        buffer = {
            "input_ids": [],
            "labels": [],
        }

        for input_ids, labels in tqdm(self.ds, desc="Packing dataset", dynamic_ncols=True):
            buffer["input_ids"].extend(input_ids)
            buffer["labels"].extend(labels)

            # If buffer has reached max_seq_len, append packed sample
            while len(buffer["input_ids"]) > self.max_seq_len:
                self.samples.append(
                    {k: v[: self.max_seq_len] for k, v in buffer.items()}
                )
                buffer = {k: v[self.max_seq_len :] for k, v in buffer.items()}
                assert len(buffer["input_ids"]) == len(buffer["labels"])
                if self.max_rows is not None and len(self.samples) >= self.max_rows:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self.samples[index]["input_ids"], self.samples[index]["labels"]
