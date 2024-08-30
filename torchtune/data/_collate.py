# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Dict, List, Tuple, Union

import torch

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX


def left_pad_sequence(
    sequences: List[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0,
) -> torch.Tensor:
    """
    This function is identical to :func:`torch.nn.utils.rnn.pad_sequence`, but
    instead pads a list of variable length Tensors from the left to the length
    of the longest sequence.

    Example:
        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([4, 5, 6, 7])
        >>> c = torch.tensor([8, 9, 10, 11, 12])
        >>> left_pad_sequence([a, b, c], padding_value=0)
        tensor([[ 0,  0,  1,  2,  3],
                [ 0,  4,  5,  6,  7],
                [ 8,  9, 10, 11, 12]])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (List[torch.Tensor]): list of variable length sequences.
        batch_first (bool): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise. Default False.
        padding_value (float): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """
    return pad_sequence(
        map(lambda x: torch.flip(x, dims=[0]), sequences),
        batch_first=batch_first,
        padding_value=padding_value,
    ).flip(dims=[1])


def padded_collate(
    batch: List[Dict[str, List[int]]],
    *,
    pad_fn: Callable,
    keys_to_pad: List[str],
    padding_idx: Union[int, Dict[str, int]],
):
    """
    A generic padding collation function which pads ``keys_to_pad`` entries in a
    batch of sequences with the given `pad_fn` to the maximum sequence length for
    each entry in the batch. Any keys in ``batch`` which are not in ``keys_to_pad``
    are discarded.

    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries containing inputs.
        pad_fn (Callable): padding function to apply to ``keys_to_pad`` batch elements.
            This should have an identical signature to :func:`torch.nn.utils.rnn.pad_sequence`, e.g.
            :func:`torchtune.data.left_pad_sequence`.
        keys_to_pad (List[str]): Batch element keys to apply padding to. Should be a subset
            of keys in the batch.
        padding_idx (Union[int, Dict[str, int]]): Either a single integer padding value to apply to all
            ``keys_to_pad`` elements, or a mapping with keys identical to ``keys_to_pad`` with per-key
            padding values.

    Returns:
        torch.Tensor: The padded tensor of input ids with shape [batch_size, max_seq_len].

    Raises:
        ValueError: if ``keys_to_pad`` is empty, or is not a list, or is not a subset of keys in the batch.
        ValueError: if ``padding_idx`` is provided as a dictionary, but the keys are not identical to
            ``keys_to_pad``.

    Example:
        >>> a = [1, 2, 3]
        >>> b = [4, 5, 6, 7]
        >>> c = [8, 9, 10, 11, 12]
        >>> batch = [
        >>>     {"tokens": a, "labels": [1]},
        >>>     {"tokens": b, "labels": [3]},
        >>>     {"tokens": c, "labels": [0]},
        >>> ]
        >>> padded_collate(
        >>>     batch,
        >>>     pad_fn=left_pad_sequence,
        >>>     keys_to_pad=["tokens"],
        >>>     padding_idx=-10
        >>> )
        {
            'labels': tensor([[1],
                            [3],
                            [0]]),
            'tokens': tensor([[-10, -10,   1,   2,   3],
                            [-10,   4,   5,   6,   7],
                            [  8,   9,  10,  11,  12]])
        }
    """
    if not isinstance(keys_to_pad, list) or not keys_to_pad:
        raise ValueError(
            f"keys_to_pad should be a list of strings with at least one element, but found {keys_to_pad}!"
        )
    if isinstance(padding_idx, dict):
        if isinstance(padding_idx, dict) and not set(padding_idx.keys()) == set(
            keys_to_pad
        ):
            raise ValueError(
                f"padding_idx was provided as a dictionary, but the keys ({padding_idx.keys()}) "
                f"are not the same as keys_to_pad ({keys_to_pad})"
            )
        if not set(keys_to_pad) <= set(batch[0].keys()):
            raise ValueError(
                "keys_to_pad should be a subset of keys in the batch, but found "
                f"{set(keys_to_pad)} and {set(batch[0].keys())}, respectively."
            )

    output_dict = {}
    for k in keys_to_pad:
        output_dict[k] = pad_fn(
            [torch.tensor(x[k]) for x in batch],
            batch_first=True,
            padding_value=padding_idx[k]
            if isinstance(padding_idx, dict)
            else padding_idx,
        )
    return output_dict


def padded_collate_sft(
    batch: List[Dict[str, List[int]]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> Dict[str, torch.Tensor]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors.

    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries containing input, label pairs.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.

    Returns:
        Dict[str, torch.Tensor]: Collated input and label tensors.

    Example:
        >>> token_pairs = [
        >>>    {"tokens": [1, 2, 3], "labels": [4, 5, 6]},
        >>>    {"tokens": [7,], "labels": [10,]},
        >>> ]
        >>> collated = padded_collate(
        >>>    batch=token_pairs,
        >>>    padding_idx=padding_idx,
        >>>    ignore_idx=ignore_idx,
        >>> )
        >>> collated["tokens"]
        >>> tensor([[1, 2, 3], [7, 0, 0]])
        >>> collated["labels"]
        >>> tensor([[4, 5, 6], [10, -100, -100]])
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
    return {"tokens": input_ids.long(), "labels": labels.long()}


def padded_collate_dpo(
    batch: List[Dict[str, List[int]]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of sequences for Direct Preference Optimization (DPO).

    This function takes a batch of sequences, where each sequence is represented
    as a dictionary with multiple key-value pairs. Each key corresponds to a different
    sequence component, such as input_ids or labels.

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

    to_pad_input_ids = chosen_input_ids + rejected_input_ids
    to_pad_labels = chosen_labels + rejected_labels

    concatenated_input_ids = pad_sequence(
        to_pad_input_ids, batch_first=True, padding_value=padding_idx
    )
    concatenated_labels = pad_sequence(
        to_pad_labels, batch_first=True, padding_value=ignore_idx
    )

    return concatenated_input_ids, concatenated_labels
