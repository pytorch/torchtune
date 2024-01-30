# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import Tensor


def _copy_tensor(t: Tensor) -> Tensor:
    """
    A torch.clone-free way to clone a torch.tensor. We implement without
    torch.clone for better compatibility with copy.deepcopy.
    """
    ret_tensor = torch.empty_like(t).requires_grad_(t.requires_grad)
    with torch.no_grad():
        ret_tensor.copy_(t)

    return ret_tensor
