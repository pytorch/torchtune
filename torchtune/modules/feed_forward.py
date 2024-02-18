# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
from torch import nn, Tensor


class FeedForward(nn.Module):
    """This class implements the feed-forward network derived from Llama2.

    Args:
        linear1 (nn.Module): TO ADD
        linear2 (nn.Module): TO ADD
        linear3 (nn.Module): TO ADD
    """

    def __init__(
        self,
        linear1: nn.Module,
        linear2: nn.Module,
        linear3: nn.Module,
    ):
        super().__init__()
        self.w1 = linear1
        self.w2 = linear2
        self.w3 = linear3
        self.activation = F.silu

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.activation(self.w1(x)) * self.w3(x))
