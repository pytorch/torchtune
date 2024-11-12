# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F


def pad_dim_to_size(
    input: torch.Tensor, size: int, dim: int, *, fill: float = 0.0
) -> torch.Tensor:
    """
    Pads the given dimension of the input to the given size.

    Example:
        >>> image = torch.rand(1, 4, 4)
        >>> padded_image = pad_to_size(image, 3, dim=0)
        >>> padded_image.shape
        torch.Size([3, 4, 4])

    Args:
        input (torch.Tensor): Tensor to pad.
        size (int): Size to pad to.
        dim (int): Dimension to pad.
        fill (float): Value to fill the padded region with. Default: 0.0

    Returns:
        torch.Tensor: Padded input.
    """
    pad_size = size - input.shape[dim]
    assert (
        pad_size >= 0
    ), f"Tensor input shape {input.shape[dim]} is larger than given size {size}"

    # Set up 0 padding for the entire tensor.
    # Padding is in order W*H*C*N, with front and back for each dim.
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    padding = [0] * 2 * input.dim()
    # Find the pad_index: convert NCHW to WHCN, and only pad the back
    # (not both sides).
    pad_index = (input.dim() - dim) * 2 - 1
    padding[pad_index] = pad_size
    # Pad dim to size.
    return F.pad(input, padding, value=fill)
