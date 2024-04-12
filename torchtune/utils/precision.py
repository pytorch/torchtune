# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import (
    ContextManager,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from pkg_resources import packaging

from torch.cuda.amp import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torchao.dtypes.nf4tensor import NF4Tensor

from torchtune.utils._device import _validate_device_from_env
from torchtune.utils.logging import get_logger

log = get_logger()

PRECISION_STR_TO_DTYPE: Dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def _set_float32_precision(precision: str = "high") -> None:
    """Sets the precision of float32 matrix multiplications and convolution operations.

    For more information, see the PyTorch docs:
    - https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    - https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.allow_tf32

    Args:
        precision (str): The setting to determine which datatypes to use for matrix multiplication and convolution operations.
    """
    if not torch.cuda.is_available():  # Not relevant for non-CUDA devices
        return
    # set precision for matrix multiplications
    torch.set_float32_matmul_precision(precision)
    # set precision for convolution operations
    if precision == "highest":
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True


def list_dtypes() -> List[str]:
    """
    Return a list of supported dtypes for finetuning.
    """
    return list(PRECISION_STR_TO_DTYPE.keys())


def verify_bf16_support():
    return (
        torch.cuda.is_available()
        and torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and torch.distributed.is_nccl_available()
        and torch.cuda.nccl.version() >= (2, 10)
    )


def get_dtype(
    dtype: Optional[str] = None, device: Optional[torch.device] = None
) -> torch.dtype:
    """Get the torch.dtype corresponding to the given precision string.

    NOTE: If bf16 precision is requested with a CUDA device, we verify whether the device indeed supports
        bf16 kernels. If not, the dtype returned is `torch.float32`.

    Args:
        dtype (Optional[str]): The precision dtype. Default: ``None``, in which we default to torch.float32
        device (Optional[torch.device]): Device in use for training. Only CUDA and CPU
            devices are supported. If a CUDA device is passed in, additional checking is done
            to ensure that the device supports the requested precision. Default: ``None``, in which case
            a CUDA device is assumed.
    Raises:
        ValueError: if precision isn't supported by the precision utils

    Returns:
        torch.dtype: The corresponding torch.dtype.

    """

    # None defaults to float32
    if dtype is None:
        return torch.float32

    # Convert to torch.dtype
    torch_dtype = PRECISION_STR_TO_DTYPE.get(dtype, dtype)

    # dtype must be one of the supported precisions
    if torch_dtype not in PRECISION_STR_TO_DTYPE.values():
        raise ValueError(
            f"Dtype {torch_dtype} must be one of {', '.join(list_dtypes())} for finetuning."
        )

    # TODO (rohan-varma): prefer to use get_default_device() here to figure out whether user is training on
    # CPU or GPU, but it is not supported in versions of torch we test.
    if (
        torch_dtype == torch.bfloat16
        and device != torch.device("cpu")
        and not verify_bf16_support()
    ):
        log.info("BF16 not supported on this hardware. Setting dtype to float32")
        torch_dtype = torch.float32

    return torch_dtype


def get_gradient_scaler(fsdp: bool = False) -> Union[GradScaler, ShardedGradScaler]:
    """
    Returns a gradient scaler for mixed-precision training.

    Args:
        fsdp (bool): Whether FSDP is being used for training, in which case a shard-aware gradient scaler is returned.

    Returns:
        Union[GradScaler, ShardedGradScaler]: Gradient scaler object
    """
    return GradScaler(enabled=True) if not fsdp else ShardedGradScaler(enabled=True)


def get_autocast(dtype: torch.dtype, device: torch.device) -> ContextManager:
    """
    Intelligently determines, based on the dtype if mixed precision training is supported and
    returns the builtin torch.autocast if applicable.

    Reference: https://pytorch.org/docs/stable/amp.html#torch.autocast

    Args:
        dtype (torch.dtype): dtype used to determine if mixed precision training is used.
        device (torch.device): Pytorch device.
    Returns:
        Autocast manager object if using half precision, otherwise null context
    """
    manager = None
    if dtype in (torch.float16, torch.bfloat16):
        # Note some devices do not support autocasting, and will raise an error.
        _validate_device_from_env(device)
        return torch.autocast(
            device_type=device.type,
            dtype=dtype,
        )
    else:
        return contextlib.nullcontext()


@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def validate_expected_param_dtype(
    named_params: Iterable[Tuple[str, nn.Parameter]], dtype: torch.dtype
) -> None:
    """
    Validates that all input parameters have the expected dtype.

    Args:
        named_params (Iterable[Tuple[str, nn.Parameter]]): Iterable of named parameters.
        dtype (torch.dtype): Expected dtype.

    Raises:
        ValueError: If any parameter has a different dtype than `dtype`.
    """
    for name, param in named_params:
        if param.dtype != dtype:
            raise ValueError(
                f"Parameter {name} has dtype {param.dtype}, but expected {dtype}"
            )


def reparametrize_as_dtype_state_dict_post_hook(
    model: nn.Module,
    state_dict: Dict[str, Any],
    *args: Tuple[Any, ...],
    dtype: torch.dtype = torch.bfloat16,
    offload_to_cpu: bool = True,
    **kwargs: Dict[Any, Any],
):
    """
    A state_dict hook that replaces nf4 tensors with their restored
    higher-precision weight and optionally offloads the restored weight to CPU.

    This function is meant to be used with PyTorch's ``nn.Module._register_state_dict_hook``, i.e.
    >>> m = MyModule()
    >>> m._register_state_dict_hook(reparametrize_as_dtype_state_dict_post_hook)

    If the hook is registered per the above process, this hook will be called _after_ the module's
    ``state_dict`` method is called. The hook will replace all ``NF4Tensor`` instances by de-quantizing
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
