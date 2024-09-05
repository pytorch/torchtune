# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


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
    cur_size = input.shape[dim]
    pad_size = size - cur_size
    assert (
        pad_size >= 0
    ), f"Tensor input shape {cur_size} is larger than given size {size}"
    if pad_size == 0:
        return input
    shape = list(input.shape)
    shape[dim] = pad_size
    padding = torch.full(
        size=shape, fill_value=fill, dtype=input.dtype, device=input.device
    )
    return torch.cat([input, padding], dim=dim)
