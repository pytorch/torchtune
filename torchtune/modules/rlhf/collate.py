# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX


def left_padded_collate(
    batch: List[Dict[str, List[int]]],
    padding_idx: int = 0,
) -> torch.Tensor:
    """
    Pads a batch of sequences with left padding to the maximum sequence length in the batch.

    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries containing inputs.
        padding_idx (int): The padding index. Defaults to 0.

    Returns:
        torch.Tensor: The padded tensor of input ids with shape [batch_size, max_seq_len].

    Example:
        >>> padding_idx = -8
        >>> batch = [
        >>>     {"tokens": [1, 2] },
        >>>     {"tokens": [3] },
        >>>     {"tokens": [4, 5, 6, 7]},
        >>> ]
        >>> left_padded_collate(batch, padding_idx)
        >>> tensor([[-8, -8,  1,  2],
        >>>         [-8, -8, -8,  3],
        >>>         [ 4,  5,  6,  7]])

    """
    pad_toks = pad_sequence(
        [torch.tensor(x["tokens"][::-1]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    seq_idxs_rev = torch.arange(pad_toks.shape[-1] - 1, -1, -1)
    return torch.stack([tok[seq_idxs_rev] for tok in pad_toks])


def padded_collate_dpo(
    batch: List[Dict[str, List[int]]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of sequences for Direct Preference Optimization (DPO).

    This function takes a batch of sequences, where each sequence is represented
    as a dictionary with multiple key-value pairs. Each key corresponds to a different
    sequence component, such as input_ids or labels.

    This will raise:
        AssertionError: if the length of chosen_input_ids and rejected_input_ids differ.
        AssertionError: if the length of chosen_labels and rejected_labels differ.

    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries, where each dictionary
            represents a sequence with multiple components, 'chosen_input_ids',
            'chosen_labels', 'rejected_input_ids', and 'rejected_labels' are required.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing concatenated and padded
        input ids and labels.


    Example:
        >>> batch = [
        >>>    {'chosen_input_ids': [1, 2, 3], 'rejected_input_ids': [4, 5],
        >>>      'chosen_labels': [6, 7, 8], 'rejected_labels': [9, 10]},
        >>>    {'chosen_input_ids': [11, 12], 'rejected_input_ids': [13, 14, 15],
        >>>      'chosen_labels': [16, 17], 'rejected_labels': [18, 19, 20]},
        >>> ]
        >>> padded_collate_dpo(batch)
        >>> (tensor([[ 1,  2,  3],
        >>>          [11, 12,  0],
        >>>          [ 4,  5,  0],
        >>>          [13, 14, 15]]),
        >>>  tensor([[ 6,  7,  8],
        >>>          [16, 17, -100],
        >>>          [ 9, 10, -100],
        >>>          [18, 19, 20]]))
    """
    chosen_input_ids = [torch.tensor(ex["chosen_input_ids"]) for ex in batch]
    rejected_input_ids = [torch.tensor(ex["rejected_input_ids"]) for ex in batch]
    chosen_labels = [torch.tensor(ex["chosen_labels"]) for ex in batch]
    rejected_labels = [torch.tensor(ex["rejected_labels"]) for ex in batch]

    assert len(chosen_input_ids) == len(rejected_input_ids)
    assert len(chosen_labels) == len(rejected_labels)

    to_pad_input_ids = chosen_input_ids + rejected_input_ids
    to_pad_labels = chosen_labels + rejected_labels

    concatenated_input_ids = pad_sequence(
        to_pad_input_ids, batch_first=True, padding_value=padding_idx
    )
    concatenated_labels = pad_sequence(
        to_pad_labels, batch_first=True, padding_value=ignore_idx
    )

    return concatenated_input_ids, concatenated_labels
