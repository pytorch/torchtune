# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor

def expand_integer_mask(mask: Tensor) -> Tensor:
    """
    Expands an integer mask denoting individual samples packed in a single
    sequence into a diagonal tril block mask. This prevents samples from
    attending to each other.

    Args:
        mask (Tensor): Integer mask denoting packed samples of shape [b x s]

    Returns:
        Tensor: Diagonal tril block mask of shape [b x s x s]
    """
    # Count sequence lengths of individual samples
    sample_nums, seq_lens = torch.unique(mask, return_counts=True)

    blocks = []
    # For each sample, create lower triangular matrix of size of its seq len
    for seq_len in seq_lens:
        block = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        blocks.append(block)
    # Concatenate the blocks along the diagonal to create the final mask
    return torch.block_diag(*blocks)
