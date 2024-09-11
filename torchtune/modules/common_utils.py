# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import mmap
import sys
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Tuple

import torch

import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorConverter, FakeTensorMode
from torchao.dtypes.nf4tensor import NF4Tensor

_use_low_cpu_ram: bool = False


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


def _low_ram_reparametrize_as_dtype_state_dict_post_hook(
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

    This hook is similar to ``reparametrize_as_dtype_state_dict_post_hook`` but uses
    FakeTensor and mmap(2) to avoid CPU OOM on colab.

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
    # Create a state dict of FakeTensors that matches the state_dict
    mode = FakeTensorMode()
    converter = FakeTensorConverter()
    fake_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if isinstance(v, NF4Tensor):
            fake_state_dict[k] = converter.from_real_tensor(mode, v).to(dtype)
        else:
            fake_state_dict[k] = converter.from_real_tensor(mode, v)

        if offload_to_cpu:
            fake_state_dict[k] = fake_state_dict[k].cpu()

    # Create a state_dict on disk with space reserved for storage bytes
    # Then load with mmap and MAP_SHARED (can writeback to disk file)
    dest_state_dict_path = "/tmp/fake_state_dict.pt"
    with torch.serialization.skip_data(materialize_fake_tensors=True):
        torch.save(fake_state_dict, dest_state_dict_path)
    with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
        dest_state_dict = torch.load(dest_state_dict_path, mmap=True, weights_only=True)

    # Do D2H and upcast one by one and since dest_state_dict is backed by mmap --> won't OOM
    # even when there is no swap space (e.g. colab)
    for k in state_dict.keys():
        if isinstance(state_dict[k], NF4Tensor):
            dest_state_dict[k].copy_(state_dict[k].to(dtype))
        else:
            dest_state_dict[k].copy_(state_dict[k])

    # In place update original state_dict object. Although the private state dict
    # post hook supports out of place behavior, the semantic actually buggy. We eventually want
    # to use the public state_dict post hook which does not support out of place behavior.
    for k in state_dict.keys():
        state_dict[k] = dest_state_dict[k]


def _register_reparametrize_state_dict_hooks(
    module: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    offload_to_cpu: bool = True,
):
    """
    Register the reparametrize state dict hooks to the module and its submodules.

    This function is a wrapper that is meant to toggle between the low_cpu_ram
    and regular versions of the ``reparametrize_as_dtype`` state dict hooks.

    Args:
        module (nn.Module): the module to register the hooks to.
        dtype (torch.dtype): the dtype to restore the weight to. Default is ``torch.bfloat16``.
        offload_to_cpu (bool): whether to offload the restored weight to CPU. Default is ``True``.

    Raises:
        RuntimeError: If the low RAM reparametrize hook is used on Windows or an incompatible torch version.
    """
    if _use_low_cpu_ram:
        if torch.__version__ < "2.5.0.dev20240906":
            raise RuntimeError(
                "Low RAM reparametrize_as_dtype_state_dict_post_hook requires PyTorch 2.5.0.dev20240906 or later."
            )
        elif sys.platform == "win32":
            # mmap.MAP_SHARED is not supported on Windows but this change targets colab.
            raise RuntimeError(
                "Low RAM reparametrize_as_dtype_state_dict_post_hook is not supported on Windows."
            )
        else:
            hook = _low_ram_reparametrize_as_dtype_state_dict_post_hook
    else:
        hook = reparametrize_as_dtype_state_dict_post_hook
    module._register_state_dict_hook(
        partial(hook, dtype=dtype, offload_to_cpu=offload_to_cpu)
    )
