# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib.metadata import PackageNotFoundError, version

import torch
from torchao.dtypes.nf4tensor import implements as nf4_tensor_impl, to_nf4


def is_fbcode():
    return not hasattr(torch.version, "git_version")


@nf4_tensor_impl([torch.ops.aten.clone.default])
def clone(func, *args, **kwargs):
    """
    __torch_dispatch__ override that is called when cloning an NF4Tensor.
    This is implemented by creating a new NF4Tensor with the unquantized weight
    of the input tensor. Note that this is not an exact "clone" due to the loss
    in precision.
    """
    return to_nf4(args[0][0].get_original_weight())


should_define_inplace_copy = True
if not is_fbcode():
    try:
        ao_version = version("torchao")
        should_define_inplace_copy = ao_version < "0.2.0"
    # For importlib metadata, need to check nightly separately
    except PackageNotFoundError:
        ao_version = version("torchao-nightly")
        should_define_inplace_copy = ao_version < "2024.5.20"
    except Exception as e:
        raise PackageNotFoundError("Could not find torchao version") from e


if should_define_inplace_copy:
    # TorchAO have `NF4.copy_` starting from `0.2.0`
    # it's a superset of `inplace_copy` since it covers `NF4.copy_(NF4)`
    @nf4_tensor_impl([torch.ops.aten.copy_.default])
    def inplace_copy(func, *args, **kwargs):
        """
        Performs an inplace copy of an incoming tensor into the tensor
        being copied into. The inplace tensor is given by args[0][1] and the
        tensor being copied into is given by args[0][0]. The copy is performed
        by copying over all attributes. This method would have to be updated
        if additional attributes are added to NF4Tensor.
        """
        dest_tensor = args[0][0]  # tensor we are inplace copying into
        ref_tensor = to_nf4(
            args[0][1].to(dest_tensor.device)
        )  # TODO check if nf4 tensor takes in device arg
        dest_tensor.block_size = ref_tensor.block_size
        dest_tensor.n_blocks = ref_tensor.n_blocks
        dest_tensor.scaler_block_size = ref_tensor.scaler_block_size
        dest_tensor.quantized_scalers = ref_tensor.quantized_scalers
        dest_tensor.quantization_factor = ref_tensor.quantization_factor
        dest_tensor.scaler_mean = ref_tensor.scaler_mean
        dest_tensor.quantized_data = ref_tensor.quantized_data
        dest_tensor.nf4 = ref_tensor.nf4
