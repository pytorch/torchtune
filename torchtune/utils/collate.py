# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Tuple

import torch

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX


def padded_collate(
    batch: List[Dict[str, Any]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> Dict[str, torch.Tensor]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors.

    Args:
        batch (List[Dict[str, Any]]): A list of tuples containing input, label pairs.
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
        [torch.tensor(x["tokens"]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    labels = pad_sequence(
        [torch.tensor(x["labels"]) for x in batch],
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
    return {"tokens": input_ids, "labels": labels}


def padded_collate_dpo(
    batch: List[Dict[str, Any]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of sequences for Direct Preference Optimization (DPO).

    This function takes a batch of sequences, where each sequence is represented
    as a dictionary with multiple key-value pairs. Each key corresponds to a different
    sequence component, such as input_ids or labels.

    Args:
        batch (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
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


def _padded_collate_packed(
    sample: Dict[str, Any],
    max_seq_len: int,
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> Dict[str, torch.Tensor]:
    """Pad a batch of packed sequences to max sequence length in the batch, and
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
        torch.tensor(tokens), (0, max_seq_len - len(tokens)), value=padding_idx
    )
    labels = F.pad(
        torch.tensor(labels), (0, max_seq_len - len(labels)), value=ignore_idx
    )
    # For attention mask, simply use identity matrix for the pad tokens
    mask_pad = torch.eye(max_seq_len - mask.shape[0], dtype=torch.bool)
    mask = torch.block_diag(mask, mask_pad)
    # For position ids, continue to increment for pad tokens
    input_pos_pad = torch.arange(
        input_pos[-1] + 1, max_seq_len - len(input_pos) + input_pos[-1] + 1
    )
    # Do not go beyond max_seq_len - 1
    input_pos_pad = input_pos_pad.clamp(max=max_seq_len - 1)
    input_pos = torch.cat(
        [
            torch.tensor(input_pos),
            input_pos_pad,
        ]
    )

    return {
        "tokens": tokens,
        "labels": labels,
        "mask": mask,
        "input_pos": input_pos,
    }
