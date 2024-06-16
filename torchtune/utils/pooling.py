# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch


def get_last_non_masked_token(mask: torch.Tensor, dtype=torch.long) -> torch.Tensor:
    """Returns the index for the last unmasked entry for each row of a 2D boolean mask.
    Args:
        mask (torch.Tensor): Boolean mask with shape [b x s]
        dtype (torch.dtype): dtype to cast the returned idxs to
    Returns:
        Tensor: Sequence indexes logits with shape [b]
    Notation used for tensor shapes:
        - b: batch size
        - s: sequence length
    """
    # calculate per-batch-element sequence lengths by finding last valid tokens
    if mask.any():
        sequence_lengths = (~mask).sum(-1).sub(1).clip(0).to(mask.device, dtype=dtype)
    else:
        sequence_lengths = -1

    return sequence_lengths
