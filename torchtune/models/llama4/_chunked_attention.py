# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torchtune.modules.attention_utils import (
    _MaskType,
    _SUPPORTS_FLEX_ATTENTION,
    causal_mask_flex,
)

if _SUPPORTS_FLEX_ATTENTION:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask


def get_chunked_attention_mask(
    mask: Optional[_MaskType],
    chunk_size: int,
    bsz: int,
    seq_len: int,
    # Unused, but listed for consistency
    device: Optional[torch.device] = None,
) -> _MaskType:
    """ """
    # TODO: check this somewhere that doesn't get called every forward
    if not _SUPPORTS_FLEX_ATTENTION:
        raise ValueError("Local attention is only supported with flex attention.")
    if mask is None:
        mask_mod = causal_mask_flex
        q_seq_len, kv_seq_len = seq_len, seq_len
    elif isinstance(mask, BlockMask):
        mask_mod = mask.mask_mod
        q_seq_len, kv_seq_len = mask.seq_lengths
    else:
        raise ValueError("Unsupported mask type")

    def chunked_attention_mask_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ):
        # Get the chunk index of the query and key
        q_chunk = q_idx // chunk_size
        kv_chunk = kv_idx // chunk_size
        # Only allow attention within the same batch
        same_chunk = q_chunk == kv_chunk
        # Apply the original mask mod
        inner_mask = mask_mod(b, h, q_idx % chunk_size, kv_idx % chunk_size)
        return same_chunk & inner_mask

    return create_block_mask(
        chunked_attention_mask_mod,
        bsz,
        None,
        q_seq_len,
        kv_seq_len,
    )
