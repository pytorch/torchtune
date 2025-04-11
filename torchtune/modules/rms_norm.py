# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization in fp32.

    See: https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to normalize

        Returns:
            torch.Tensor: The normalized and scaled tensor having the same shape as ``x``.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.scale


def rms_norm(x: torch.Tensor, eps: float = 1e-6):
    """
    This is just a functional RMSNorm without the trainable scale parameter.

    Args:
        x (torch.Tensor): input tensor to normalize
        eps (float): small value to avoid division by zero. Default: 1e-6

    Returns:
        torch.Tensor: The normalized tensor having the same shape as ``x``.

    """
    x_fp32 = x.float()
    x_normed = (
        x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    ).type_as(x)
    return x_normed
