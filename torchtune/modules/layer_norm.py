# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torch import nn, Tensor


class LayerNorm(nn.Module):
    """
    Wrapper around torch.nn.LayerNorm to support fp16 training.

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-5
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor to normalize

        Returns:
            Tensor: The output tensor after applying LayerNorm.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x = self.layernorm(x_fp32)
        return x.type_as(x)
