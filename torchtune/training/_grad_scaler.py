# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


def scale_grads(model: nn.Module, scaler: torch.Tensor) -> None:
    """
    Utility to scale the gradients of a model.
    This is useful for gradient accumulation where we want to normalize
    the gradients by the total number of tokens seen.

    Inputs:
        model (nn.Module): model whose gradients should be scaled
        scaler (torch.Tensor): scaling factor to apply to the gradients

    Outputs:
        None (grad fields are modified in place)
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad *= scaler
