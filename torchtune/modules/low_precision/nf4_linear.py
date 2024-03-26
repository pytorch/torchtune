# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchao.dtypes.nf4tensor import NF4Tensor, to_nf4

# TODO (rohan-varma): Remove this asap after torchao side changes land to decouple
# linear_nf4 from bf16.
class _LinearNF4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: NF4Tensor):
        """Save the quantized nf4 weight for backward pass"""
        ctx.nf4_weight = weight
        return F.linear(input, weight.to(input.dtype))

    @staticmethod
    def backward(ctx, grad_output):
        """The nf4 weight will never require grad so we can just return the grad_output @ weight.get_original_weight()"""
        weight: NF4Tensor = ctx.nf4_weight
        return grad_output @ weight.to(grad_output.dtype), None


def _linear_nf4(input: torch.Tensor, weight: NF4Tensor) -> torch.Tensor:
    return _LinearNF4.apply(input, weight)


class FrozenNF4Linear(nn.Linear):
    """
    A linear layer similar to ``torch.nn.Linear`` but uses a quantized
    NF4Tensor as its weight. This class also freezes its ``weight`` parameter
    and is meant to be used as the base Linear layer for modeling
    use cases such as QLoRA where base model parameters are frozen.
    NOTE: biases are currently not supported.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        device (Optional[torch.device]): device to use for the underlying weight. If ``None``, uses the default
            device given by `torch.get_default_device()`.
        **kwargs: any additional arguments to pass to the underlying Linear layer.

    Raises:
        RuntimeError: if ``bias`` is set to ``True``
    """

    def __init__(
        self, in_dim: int, out_dim: int, device: Optional[torch.device] = None, **kwargs
    ):
        if "bias" in kwargs and kwargs.pop("bias"):
            raise RuntimeError("FrozenNF4Linear does not currently support biases!")

        super().__init__(in_dim, out_dim, device=device, bias=False, **kwargs)
        self.weight.requires_grad_(False)
        self.nf4_weight = to_nf4(self.weight.data)
        # re-register self.weight as the nf4 weight, so that the nf4 weight
        # shows up as expected in .parameters, state_dict, etc.
        self.weight = torch.nn.Parameter(self.nf4_weight, requires_grad=False)

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs linear operation with input tensor as given by `input`. Computation happens in higher
        precision, though only the nf4 weight is saved for backward for gradient computation to ensure
        additional memory is not used.
        Args:
            input (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        return _linear_nf4(input=input, weight=self.weight)
