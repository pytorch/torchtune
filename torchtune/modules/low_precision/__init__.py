# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchao.dtypes.nf4tensor import NF4Tensor

from .nf4_linear import FrozenNF4Linear


def reparametrize_as_bf16_state_dict_post_hook(
    model, state_dict, *args, offload_to_cpu: bool = True, **kwargs
):
    """
    Replaces nf4 tensors with their restored bf16 weight.
    """
    print(f"RV: replace as bf16 and offload to CPU.")
    for k, v in state_dict.items():
        if isinstance(v, NF4Tensor):
            state_dict[k] = v.get_original_weight()
            if offload_to_cpu:
                state_dict[k] = state_dict[k].cpu()


# def reparametrize_as_bf16(model, offload_to_cpu):
#     """
#     Replace
#     """
#     for name, module in model.named_modules():
#         if isinstance(module, FrozenNF4Linear):
#             module.reparametrize_as_bf16(offload_to_cpu)

__all__ = [
    "FrozenNF4Linear",
    "reparametrize_as_bf16_state_dict_post_hook"
]
