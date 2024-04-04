# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Tuple, Dict

import torch

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# TokenPair is a pair (tuple) of two lists: tokenized text inputs and labels.
TokenPair = Tuple[List[int], List[int]]


def padded_collate(
    batch: List[TokenPair],
    padding_idx: int = 0,
    ignore_idx: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors.

    Args:
        batch (List[TokenPair]): A list of tuples containing input, label pairs.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.

    Returns:
        Collated input and label tensors.

    Example:
        >>> token_pairs = [
        >>>    ([1, 2, 3], [4, 5, 6]),
        >>>    ([7,], [10,],),
        >>> ]
        >>> inputs, labels = padded_collate(
        >>>    batch=token_pairs,
        >>>    padding_idx=padding_idx,
        >>>    ignore_idx=ignore_idx,
        >>> )
        >>> inputs
        >>> tensor([[1, 2, 3], [7, 0, 0]])
        >>> labels
        >>> tensor([[4,5,6], [10,-100,-100]])
    """
    input_ids = pad_sequence(
        [torch.tensor(x[0]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    labels = pad_sequence(
        [torch.tensor(x[1]) for x in batch],
        batch_first=True,
        padding_value=ignore_idx,
    )

    input_ids_seq_len = input_ids.shape[-1]
    labels_seq_len = labels.shape[-1]

    # Hack to pad correctly and not use max_seq_len, which is costly
    if input_ids_seq_len > labels_seq_len:
        labels = F.pad(
            labels, (0, input_ids_seq_len - labels_seq_len), value=ignore_idx
        )
    elif labels_seq_len > input_ids_seq_len:
        input_ids = F.pad(
            input_ids,
            (0, labels_seq_len - input_ids_seq_len),
            value=padding_idx,
        )
    return input_ids, labels


def padded_collate_dpo(
    batch: List[TokenPair],
    padding_idx: int = 0,
    ignore_idx: int = -100,
) -> Dict[str, torch.Tensor]:
    """Pad a batch of sequences for Direct Preference Optimization (DPO).

    This function takes a batch of sequences, where each sequence is represented
    as a dictionary with multiple key-value pairs. Each key corresponds to a different
    sequence component, such as input_ids or labels. The function pads each sequence
    component independently to the longest sequence length in the batch.

    Args:
        batch (List[TokenPair]): A list of dictionaries, where each dictionary
            represents a sequence with multiple components including 'chosen_input_ids',
            'chosen_labels', 'rejected_input_ids', and 'rejected_labels'.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.

    Returns:
        dict: A dictionary containing padded sequences for each component.

    Example:
        >>> token_pairs = [
        >>>    {'chosen_input_ids': [1, 2, 3], 'chosen_labels': [4, 5, 6],
        >>>     'rejected_input_ids': [7, 8], 'rejected_labels': [9, 10]},
        >>>    {'chosen_input_ids': [11], 'chosen_labels': [12],
        >>>     'rejected_input_ids': [13, 14, 15], 'rejected_labels': [16, 17, 18]},
        >>> ]
        >>> padded_data = padded_collate_dpo(
        >>>    batch=token_pairs,
        >>>    padding_idx=padding_idx,
        >>>    ignore_idx=ignore_idx,
        >>> )
        >>> padded_data
        >>> {
        >>>     'chosen_input_ids': tensor([[ 1,  2,  3], [11,  0,  0]]),
        >>>     'chosen_labels': tensor([[ 4,  5,  6], [12, -100, -100]]),
        >>>     'rejected_input_ids': tensor([[ 7,  8,  0], [13, 14, 15]]),
        >>>     'rejected_labels': tensor([[ 9, 10, -100], [16, 17, 18]]),
        >>> }
    """
    padded_batch = {} 
    for k in batch[0].keys():
        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
        if k.endswith("_input_ids"):
            padding_value = padding_idx
        elif k.endswith("_labels"):
            padding_value = ignore_idx
        padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)

    return padded_batch