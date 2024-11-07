# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

import torch.nn as nn
from torchao.dtypes.nf4tensor import linear_nf4, to_nf4


class FrozenNF4Linear(nn.Linear):
    """
    A linear layer similar to ``torch.nn.Linear`` but uses a quantized
    NF4Tensor as its weight. This class also freezes its ``weight`` parameter
    and is meant to be used as the base Linear layer for modeling
    use cases such as QLoRA where base model parameters are frozen.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        device (Optional[torch.device]): device to use for the underlying weight. If ``None``, uses the default
            device given by `torch.get_default_device()`.
        bias (bool): whether to include bias in the linear layer. Default: False
        **kwargs: any additional arguments to pass to the underlying Linear layer.

    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: Optional[torch.device] = None,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(in_dim, out_dim, device=device, bias=bias, **kwargs)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        nf4_weight = to_nf4(self.weight)
        # re-register self.weight as the nf4 weight, so that the nf4 weight
        # shows up as expected in .parameters, state_dict, etc.
        torch.utils.swap_tensors(
            self.weight, torch.nn.Parameter(nf4_weight, requires_grad=False)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Runs linear operation with input tensor as given by `input`. Computation happens in higher
        precision, though only the nf4 weight is saved for backward for gradient computation to ensure
        additional memory is not used.
        Args:
            input (torch.Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        out = linear_nf4(input=input, weight=self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out
