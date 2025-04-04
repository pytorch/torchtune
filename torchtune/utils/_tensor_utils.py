# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch


def chunk(tensor: torch.Tensor, chunks: int, dim: int = 0) -> Tuple[torch.Tensor]:
    """
    Attempts to split a tensor evenly into the exact number of chunks. Each chunk is a view of the input tensor.

    Standard torch.chunk/split may split tensor into a tuple with less elements than specified.
    That causes crash by timeout since other nodes in sharding wait for exact amount of tensors.
    This function splits tensor into the exact number of chunks.

    Args:
        tensor (torch.Tensor): The tensor to chunk.
        chunks (int): Number of chunks to return in a tuple.
        dim (int): Dimension along which to split the

    Returns:
        Tuple[torch.Tensor]: Chunked tensor.

    Example:
        >>> tensor = torch.rand(2, 49, 128)
        >>> chunks = chunk(tensor, 8, dim=1)
        >>> len(chunks)
        8
        >>> [x.size()[1] for x in chunks]
        [6, 6, 6, 6, 6, 6, 6, 7]

    """
    base, reminder = divmod(tensor.size(dim), chunks)
    return tensor.split([base] * (chunks - 1) + [base + reminder], dim=dim)
