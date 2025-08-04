# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torchtune.modules.attention_utils import _MaskType


def get_sliding_attention_mask(
    mask: Optional[_MaskType],
    sliding_window_size: int,
    bsz: int,
    seq_len: int,
    device: Optional[torch.device] = None,
) -> _MaskType:
    """
    Args:
        mask (Optional[_MaskType]): Mask to apply to the attention scores.
        sliding_window_size (int): Sliding window size to apply to the attention mask.
        bsz (int): Batch size. Argument is unused, but listed for consistency.
        seq_len (int): Sequence length.
        device (Optional[torch.device]): Device to use for the mask. Defaults to None.

    Returns:
        A tensor mask that applies sliding window masking.

    Raises:
        ValueError: If the input mask is not a Tensor
    """

    if mask is None:
        mask = torch.tril(
            torch.ones(size=(bsz, seq_len, seq_len), dtype=torch.bool).to(device)
        )

    if not isinstance(mask, torch.Tensor):
        raise ValueError(
            f"For non-flex attention, mask must be a Tensor. Got: {type(mask)}"
        )

    all_ones = torch.ones_like(mask, dtype=torch.bool)
    sliding_mask = torch.triu(all_ones, -1 * sliding_window_size + 1) & torch.tril(
        all_ones, sliding_window_size - 1
    )
    mask = mask & sliding_mask

    return mask
