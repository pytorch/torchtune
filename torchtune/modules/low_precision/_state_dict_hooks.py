# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import torch.nn as nn
from torchao.dtypes.nf4tensor import NF4Tensor


def reparametrize_as_bf16_state_dict_post_hook(
    model: nn.Module,
    state_dict: Dict[str, Any],
    *args: Tuple[Any, ...],
    offload_to_cpu: bool = True,
    **kwargs: Dict[Any, Any],
):
    """
    A state_dict hook that replaces nf4 tensors with their restored
    bf16 weight and optionally offloads the restored weight to CPU.

    This function is meant to be used with PyTorch's ``nn.Module._register_state_dict_hook``, i.e.
    >>> m = MyModule()
    >>> m._register_state_dict_hook(reparametrize_as_bf16_state_dict_post_hook)

    If the hook is registered per the above process, this hook will be called _after_ the module's
    ``state_dict`` method is called. The hook will replace all ``NF4Tensor`` instances by unquantizing
    them to the original dtype, and optionally offload the restored weight to CPU.

    Args:
        model (nn.Module): the model to take ``state_dict()`` on
        state_dict (Dict[str, Any]): the state dict to modify
        *args (Tuple[Any, ...]): Unused args passed when running this as a state_dict hook.
        offload_to_cpu (bool): whether to offload the restored weight to CPU
        **kwargs (Dict[Any, Any]): Unused keyword args passed when running this as a state_dict hook.
    """
    for k, v in state_dict.items():
        if isinstance(v, NF4Tensor):
            state_dict[k] = v.get_original_weight()
            if offload_to_cpu:
                state_dict[k] = state_dict[k].cpu()
