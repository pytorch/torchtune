# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any

from torch import nn, Tensor


class Fp32LayerNorm(nn.LayerNorm):
    """
    Wrapper around nn.functional.layer_norm to support mixed-precision training.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: The normalized output tensor.
        """
        output = nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)
