# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import torch
from torchtune.utils.tensor_utils import _copy_tensor


class TestTensorUtils:
    def test_copy_tensor(self):
        x = torch.rand(10, 10)
        y = _copy_tensor(x)
        assert x.shape == y.shape
        assert x.dtype == y.dtype
        assert x.device == y.device
        assert x.requires_grad == y.requires_grad
        assert x.grad_fn == y.grad_fn
        assert x.is_leaf == y.is_leaf
