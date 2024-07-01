# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import torch

import torch.nn as nn
from torchao.dtypes.nf4tensor import NF4Tensor


def reparametrize_as_dtype_state_dict_post_hook(
    model: nn.Module,
    state_dict: Dict[str, Any],
    *args: Tuple[Any, ...],
    dtype: torch.dtype = torch.bfloat16,
    offload_to_cpu: bool = True,
    **kwargs: Dict[Any, Any],
):
    """
    A state_dict hook that replaces NF4 tensors with their restored
    higher-precision weight and optionally offloads the restored weight to CPU.
    Use this hook to avoid increased peak GPU memory usage during checkpoint
    save when training with QLoRA.

    This function is meant to be used with PyTorch's ``nn.Module._register_state_dict_hook``, i.e.

    >>> m = MyModule()
    >>> m._register_state_dict_hook(reparametrize_as_dtype_state_dict_post_hook)

    If the hook is registered per the above process, this hook will be called _after_ the module's
    ``state_dict`` method is called. The hook will replace all ``NF4Tensor`` instances by unquantizing
    them to the original dtype, and optionally offload the restored weight to CPU.

    Args:
        model (nn.Module): the model to take ``state_dict()`` on
        state_dict (Dict[str, Any]): the state dict to modify
        *args (Tuple[Any, ...]): Unused args passed when running this as a state_dict hook.
        dtype (torch.dtype): the dtype to restore the weight to. Default is ``torch.bfloat16``.
        offload_to_cpu (bool): whether to offload the restored weight to CPU. Default is ``True``.
        **kwargs (Dict[Any, Any]): Unused keyword args passed when running this as a state_dict hook.
    """
    for k, v in state_dict.items():
        if isinstance(v, NF4Tensor):
            state_dict[k] = v.to(dtype)
            if offload_to_cpu:
                state_dict[k] = state_dict[k].cpu()

def slice_str_to_array(slice_str, length):
    # Parse the slice string
    parts = slice_str.split(':')
    start, end, step = None, None, None

    if len(parts) == 1 and parts[0] != '':
        start = int(parts[0])
    elif len(parts) == 2:
        start = int(parts[0]) if parts[0] != '' else None
        end = int(parts[1]) if parts[1] != '' else None
    elif len(parts) == 3:
        start = int(parts[0]) if parts[0] != '' else None
        end = int(parts[1]) if parts[1] != '' else None
        step = int(parts[2]) if parts[2] != '' else None

    # Create a boolean array based on the slice
    result = [False] * length
    slice_indices = range(start if start is not None else 0,
                          end if end is not None else length,
                          step if step is not None else 1)

    for i in slice_indices:
        if 0 <= i < length:
            result[i] = True

    return result
