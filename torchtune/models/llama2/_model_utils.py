# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional


def scale_hidden_dim_for_mlp(
    dim: int, multiple_of: int = 256, dim_multiplier: Optional[float] = None
) -> int:
    """Scale hidden dimension for MLP to keep number of parameters and computation constant.

    Args:
        dim (int): Input dimension.
        multiple_of (int): Round scaled dimension to nearest multiple of `multiple_of` for clean computation.
        dim_multiplier (Optional[float]): Multiplier for scaling hidden dimension. If ``None``, do not multiply
            hidden dimension. Default: ``None``. Note that if the result of the multiplication is a floating point,
            it is rounded down to the nearest integer.

    Returns:
        Scaled hidden dimension.
    """
    # Scale hidden dimension by (2/3)4d for SwiGLU to keep number of
    # parameters and computation constant
    hidden_dim = 4 * int(2 * dim / 3)
    if dim_multiplier is not None:
        hidden_dim = int(dim_multiplier * hidden_dim)
    # Round hidden dimension to nearest multiple of `multiple_of`
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim
