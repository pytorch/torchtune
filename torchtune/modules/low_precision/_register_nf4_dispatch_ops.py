# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchao.dtypes.nf4tensor import implements as nf4_tensor_impl, to_nf4


@nf4_tensor_impl([torch.ops.aten.clone.default])
def clone(func, *args, **kwargs):
    """
    __torch_dispatch__ override that is called when cloning an NF4Tensor.
    This is implemented by creating a new NF4Tensor with the unquantized weight
    of the input tensor. Note that this is not an exact "clone" due to the loss
    in precision.
    """
    return to_nf4(args[0][0].get_original_weight())
