# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import mmap
import sys
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Generator, Optional, Tuple
from warnings import warn

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


@contextlib.contextmanager
def disable_kv_cache(model: nn.Module) -> Generator[None, None, None]:
    """
    This context manager temporarily disables KV-cacheing on a given model, which must already
    already have KV-caches setup. All forward passes using the model within this context manager
    will not use KV-caches.

    KV-caches will be disabled when entering the context manager, and will be enabled upon exit,
    without being modified.

    This is useful in cases where we might wish to alternate between model calls which use KV-cacheing,
    and model calls which do not use KV-cacheing, without the additional overhead of deleting and setting caches up
    every time.

    Example:
        >>> from torchtune.models.llama3_2 import llama3_2_1b
        >>> from torchtune.modules import disable_kv_cache
        >>> import torch
        >>> model = llama3_2_1b()
        >>> # setup caches
        >>> model.setup_caches(batch_size=1,
        >>>                     dtype=torch.float32,
        >>>                     decoder_max_seq_len=1024)
        >>> print(model.caches_are_setup())
        True
        >>> print(model.caches_are_enabled())
        True
        >>> print(model.layers[0].attn.kv_cache)
        KVCache()
        >>> # now temporarily disable caches
        >>> with disable_kv_cache(model):
        >>>     print(model.caches_are_setup())
        True
        >>>     print(model.caches_are_enabled())
        False
        >>>     print(model.layers[0].attn.kv_cache)
        KVCache()
        >>> # caches are now re-enabled, and their state is untouched
        >>> print(model.caches_are_setup())
        True
        >>> print(model.caches_are_enabled())
        True
        >>> print(model.layers[0].attn.kv_cache)
        KVCache()

    Args:
        model (nn.Module): model to disable KV-cacheing for.

    Yields:
        None: Returns control to the caller with KV-caches disabled on the given model.

    Raises:
        ValueError: If the model does not have caches setup. Use :func:`~torchtune.modules.TransformerDecoder.setup_caches` to
            setup caches first.
    """
    if not model.caches_are_setup():
        raise ValueError(
            "Model caches must be setup before calling disable_kv_cache! "
            "Please use model.setup_caches() to setup model caches."
        )
    if not model.caches_are_enabled():
        warn(
            "You are using disable_kv_cache with a model that does not "
            "have caches enabled. This is a no-op and the expected behaviour "
            "may not occur."
        )
    for module in model.modules():
        if hasattr(module, "kv_cache") and callable(module.kv_cache):
            module.cache_enabled = False
    try:
        yield
    finally:
        for module in model.modules():
            if hasattr(module, "kv_cache") and callable(module.kv_cache):
                module.cache_enabled = True


@contextlib.contextmanager
def local_kv_cache(
    model: nn.Module,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    encoder_max_seq_len: Optional[int] = None,
    decoder_max_seq_len: Optional[int] = None,
) -> Generator[None, None, None]:
    """
    This context manager temporarily enables KV-cacheing on a given model, which does not
    already have KV-caches setup. All forward passes using the model within this context manager
    will use KV-caches.

    KV-caches will be set-up with the given ``batch_size``, ``dtype``, and ``max_seq_len`` when
    entering the context manager, and will be deleted on exit.

    Example:
        >>> from torchtune.models.llama3_2 import llama3_2_1b
        >>> from torchtune.modules import local_kv_cache
        >>> import torch
        >>> model = llama3_2_1b()
        >>> print(model.caches_are_setup())
        False
        >>> print(model.caches_are_enabled())
        False
        >>> print(model.layers[0].attn.kv_cache)
        None
        >>> # entering cacheing mode
        >>> with local_kv_cache(model,
        >>>                     batch_size=1,
        >>>                     device=torch.device("cpu"),
        >>>                     dtype=torch.float32,
        >>>                     decoder_max_seq_len=1024):
        >>>     print(model.caches_are_setup())
        True
        >>>     print(model.caches_are_enabled())
        True
        >>>     print(model.layers[0].attn.kv_cache)
        KVCache()
        >>> # exited cacheing mode
        >>> print(model.caches_are_setup())
        False
        >>> print(model.caches_are_enabled())
        False
        >>> print(model.layers[0].attn.kv_cache)
        None

    Args:
        model (nn.Module): model to enable KV-cacheing for.
        batch_size (int): batch size for the caches.
        device (torch.device): device to setup caches on. this should be the same device
            the model is on.
        dtype (torch.dtype): dtype for the caches.
        encoder_max_seq_len (Optional[int]): maximum encoder cache sequence length.
        decoder_max_seq_len (Optional[int]): maximum decoder cache sequence length.

    Yields:
        None: Returns control to the caller with KV-caches setup and enabled on the given model.

    Raises:
        ValueError: If the model already has caches setup.
            You may use :func:`~torchtune.modules.common_utils.delete_kv_caches` to delete existing caches.
    """
    if model.caches_are_setup():
        raise ValueError(
            "Model caches must be not setup prior to entering this context manager! "
            "Please use delete_kv_caches(model) to delete model caches."
        )
    # ensure caches are setup on the same device as the model
    with device:
        model.setup_caches(
            batch_size,
            dtype,
            encoder_max_seq_len=encoder_max_seq_len,
            decoder_max_seq_len=decoder_max_seq_len,
        )
    try:
        yield
    finally:
        delete_kv_caches(model)


def delete_kv_caches(model: nn.Module):
    """
    Deletes KV caches from all attention layers in a model,
    and also ensures ``cache_enabled`` is set to False.

    Example:
        >>> from torchtune.models.llama3_2 import llama3_2_1b
        >>> from torchtune.modules import delete_kv_caches
        >>> import torch
        >>> model = llama3_2_1b()
        >>> model.setup_caches(batch_size=1,
        >>>                     dtype=torch.float32,
        >>>                     decoder_max_seq_len=1024)
        >>> print(model.caches_are_setup())
        True
        >>> print(model.caches_are_enabled())
        True
        >>> print(model.layers[0].attn.kv_cache)
        KVCache()
        >>> delete_kv_caches(model)
        >>> print(model.caches_are_setup())
        False
        >>> print(model.caches_are_enabled())
        False
        >>> print(model.layers[0].attn.kv_cache)
        None

    Args:
        model (nn.Module): model to enable KV-cacheing for.

    Raises:
        ValueError: if this function is called on a model which does not have
            caches setup. Use :func:`~torchtune.modules.TransformerDecoder.setup_caches` to
            setup caches first.
    """
    if not model.caches_are_setup():
        raise ValueError(
            "You have tried to delete model caches, but model.caches_are_setup() "
            "is False! Please setup caches on the model first."
        )
    for module in model.modules():
        if hasattr(module, "kv_cache") and callable(module.kv_cache):
            module.cache_enabled = False
            module.kv_cache = None
