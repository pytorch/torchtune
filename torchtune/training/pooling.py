# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch


def get_unmasked_sequence_lengths(mask: torch.Tensor) -> torch.Tensor:
    """
    Returns the sequence lengths for each batch element, excluding masked tokens.

    Args:
        mask (torch.Tensor): Boolean mask with shape [b x s], where True indicates a value to be masked out
            This is usually a mask for padding tokens, where True indicates a padding token.

    Returns:
        Tensor: Sequence indices logits with shape [b]

    Shape notation:
        - b = batch size
        - s = sequence length

    Example:
        >>> input_ids = torch.tensor([
        ...        [2, 4, 0, 0],
        ...        [2, 4, 6, 0],
        ...        [2, 4, 6, 9]
        ...    ])
        >>> mask = input_ids == 0
        >>> mask
        tensor([[False, False,  True,  True],
                [False, False, False,  True],
                [False, False, False, False]])
        >>> get_unmasked_sequence_lengths(mask)
        tensor([1, 2, 3])

    """
    # calculate per-batch-element sequence lengths by finding last valid tokens
    if mask.any():
        sequence_lengths = (
            (~mask).sum(-1).sub(1).clip(0).to(mask.device, dtype=torch.long)
        )
    else:
        sequence_lengths = torch.full(
            (mask.shape[0],), mask.shape[1] - 1, dtype=torch.long, device=mask.device
        )

    return sequence_lengths
