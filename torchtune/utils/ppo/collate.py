# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


def left_padded_collate(
    batch: List[Dict[str, List[int]]], max_seq_len: int, padding_idx: int = 0
) -> torch.Tensor:
    """
    Pads a batch of sequences with left padding to the maximum sequence length in the batch.
    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries containing inputs.
        max_seq_len (int): The maximum sequence length to pad to.
        padding_idx (int): The padding index. Defaults to 0.
    Returns:
        torch.Tensor: The padded tensor of input ids with shape [batch_size, max_seq_len].

    """
    pad_toks = pad_sequence(
        [torch.tensor(x["tokens"][::-1]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    seq_idxs_rev = torch.arange(max_seq_len - 1, -1, -1)
    return torch.stack([tok[seq_idxs_rev] for tok in pad_toks])
