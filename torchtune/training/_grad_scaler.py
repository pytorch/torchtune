# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


def scale_grads(m: nn.Module, scaler: torch.Tensor) -> None:
    for p in m.parameters():
        if p.grad is not None:
            p.grad *= scaler
