# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch

from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask
from torchtune.utils._version import torch_version_ge

if torch_version_ge("2.5.0"):
    from torch.nn.attention.flex_attention import BlockMask

    _MaskType = Optional[Union[Tensor, BlockMask]]
else:
    _MaskType = Optional[Tensor]


def packed_block_causal_mask(
    document_ids: Tensor,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    device: torch.device,
) -> _MaskType:
    document_ids = document_ids.to(device)

    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
        padding_mask = (
            (document_ids[b, q_idx] != -1) & (document_ids[b, kv_idx] != -1)
        ) | (q_idx == kv_idx)
        return causal_mask & document_mask & padding_mask

    if torch_version_ge("2.5.0"):
        return create_block_mask(
            mask_mod, batch_size, num_heads, seq_len, seq_len, device=device
        )
    else:
        return None
