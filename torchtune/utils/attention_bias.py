# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch

from torch import Tensor
from torchtune.utils._version import torch_version_ge

if torch_version_ge("2.5.0"):
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    _MaskType = Optional[Union[Tensor, BlockMask]]
else:
    _MaskType = Optional[Tensor]


def _get_document_ids_from_seq_lens(
    seq_lens: Tensor,
) -> Tensor:
    """
    Convert a batch tensor of seq lens into integer IDs denoting sample ownership.
    For example, seq_lens = [2, 3, 1] would return [0, 0, 1, 1, 1, 2].

    Args:
        seq_lens (Tensor): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences across packs.

    Returns:
        Tensor: Document IDs of shape (batch_size, max_seq_len).
    """
    batch_size = seq_lens.shape[0]
    batch_document_ids = []
    for sample_idx in range(batch_size):
        # We assume seq lens sum to max seq lens, so document_ids should be of
        # shape (max_seq_len, )
        document_ids = torch.cat(
            [
                torch.full((seq_len,), i, dtype=torch.long, device=seq_lens.device)
                for i, seq_len in enumerate(seq_lens[sample_idx])
            ]
        )
        batch_document_ids.append(document_ids)
    batch_document_ids = torch.stack(batch_document_ids)
    return batch_document_ids


def create_block_causal_mask(seq_lens: Tensor) -> Tensor:
    """
    Given a batch tensor of seq lens defining the lengths of samples in each pack,
    Construct a 2D block causal mask for each pack in the batch. For example, if
    a single sample's seq_lens is [3, 2, 1], the mask would be::

        mask = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]

    Args:
        seq_lens (Tensor): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences across packs.

    Returns:
        Tensor: Block causal mask of shape (batch_size, max_seq_len, max_seq_len).
    """
    batch_block_attn_masks = []
    batch_size = seq_lens.shape[0]
    for sample_idx in range(batch_size):
        block_attn_masks = [
            torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=seq_lens.device)
            )
            for i, seq_len in enumerate(seq_lens[sample_idx])
        ]

        batch_block_attn_masks.append(torch.block_diag(*block_attn_masks))
    return torch.stack(batch_block_attn_masks)


def packed_block_causal_mask(
    seq_lens: Tensor,
    device: torch.device,
) -> _MaskType:
    """
    Create a block causal document mask for a batch of packed sequences. If on
    torch version >= 2.5.0, this is done by creating a mask_mod function with the
    block causal logic and passing this into :func:`torch.nn.attention.flex_attention.create_block_mask`.
    The resultant BlockMask is a compressed representation of the full block causal
    mask. If on an older version, a standard 2D block causal mask is created and returned.

    Args:
        seq_lens (Tensor): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences across packs.
        device (torch.device): Device to create the mask on.

    Returns:
        _MaskType: BlockMask or Tensor if torch version < 2.5.0.
    """
    if torch_version_ge("2.5.0"):
        document_ids = _get_document_ids_from_seq_lens(seq_lens)
        batch_size, max_seq_len = document_ids.shape
        document_ids = document_ids.to(device)

        def mask_mod(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
            return causal_mask & document_mask

        return create_block_mask(
            mask_mod,
            batch_size,
            1,
            max_seq_len,
            max_seq_len,
            device=device,
        )
    else:
        return create_block_causal_mask(seq_lens=seq_lens).to(device)
