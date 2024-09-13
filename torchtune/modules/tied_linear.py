# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class TiedLinear:
    """
    A tied linear layer, without bias, that shares the same weight as another linear layer.
    This is useful for models that use tied weights, such as qwen and gemma.
    It requires as input an nn.Module, instead of the weight of the module, so it
    can work with FSDP. Otherwise, the memory reference will be lost after FSDP is applied.

    Args:
        tied_module (nn.Module): The module whose weight is shared. Only
            the weight is used. The bias is ignored.
    Raises:
        AttributeError: If the provided module does not have an attribute 'weight'.
    """

    def __init__(self, tied_module: nn.Module):
        self.tied_module = tied_module
        if not hasattr(tied_module, "weight"):
            raise AttributeError(
                "Provided module does not have attribute 'weight'. Please check your tied_module."
            )

    def __call__(self, x: torch.tensor) -> torch.tensor:
        return F.linear(x, self.tied_module.weight)
