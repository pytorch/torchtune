# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
from torch import nn, Tensor


class FeedForward(nn.Module):
    """This class implements the feed-forward network of LLaMA.
    Notably, this utilizes a variant of GLU called SwiGLU, a combination of Swish
    and Gated Linear Units, formulated in https://arxiv.org/pdf/2002.05202.pdf (Shazeer 2020).

    Reference implementation (used for correctness verfication) can be
    found here: https://github.com/facebookresearch/llama/model.py#L307.

    Args:
        linear1 (nn.Module):
        linear2 (nn.Module):
        linear3 (nn.Module):
        activation (Union[Callable, nn.Module]):
    """

    def __init__(
        self,
        linear1: nn.Module,
        linear2: nn.Module,
        linear3: nn.Module,
        activation: Union[Callable, nn.Module],
    ):
        super().__init__()
        self.w1 = linear1
        self.w2 = linear2
        self.w3 = linear3
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.activation(self.w1(x)) * self.w3(x))
