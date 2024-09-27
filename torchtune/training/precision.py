# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Dict, Generator, Iterable, Optional, Tuple

import torch

from torchtune.utils import get_logger

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


def verify_bf16_support() -> bool:
    """
    Check that bf16 is available on this hardware. Requirements:
        - CUDA is available and supports bf16
            - CUDA version >= 11
            - CUDA compute capability >= 8
        - NCCL is available and version >= 2.10
        - MPS is available and torch was built with MPS

    Returns:
        bool: True if bf16 is available, False otherwise.

    """
    cuda_support = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and torch.distributed.is_nccl_available()
        and torch.cuda.nccl.version() >= (2, 10)
    )
    mps_support = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    return cuda_support or mps_support


def get_dtype(
    dtype: Optional[str] = None, device: Optional[torch.device] = None
) -> torch.dtype:
    """Get the torch.dtype corresponding to the given precision string. If no string is passed,
    we will default to torch.float32.

    Note:
        If bf16 precision is requested with a CUDA device, we verify whether the device indeed supports
        bf16 kernels. If not, a ``RuntimeError`` is raised.

    Args:
        dtype (Optional[str]): The precision dtype. Default: ``None``, in which we default to torch.float32
        device (Optional[torch.device]): Device in use for training. Only CUDA and CPU
            devices are supported. If a CUDA device is passed in, additional checking is done
            to ensure that the device supports the requested precision. Default: ``None``, in which case
            a CUDA device is assumed.
    Raises:
        ValueError: if precision isn't supported by the library
        RuntimeError: if bf16 precision is requested but not available on this hardware.

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
            f"Dtype {torch_dtype} must be one of {', '.join(list(PRECISION_STR_TO_DTYPE.keys()))} for finetuning."
        )

    if (
        torch_dtype == torch.bfloat16
        and device != torch.device("cpu")
        and not verify_bf16_support()
    ):
        raise RuntimeError(
            "bf16 precision was requested but not available on this hardware. Please use fp32 precision instead."
        )

    return torch_dtype


@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (torch.dtype): The desired default dtype inside the context manager.

    Returns:
        ContextManager: context manager for setting default dtype.

    Example:
        >>> with set_default_dtype(torch.bfloat16):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16


    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def validate_expected_param_dtype(
    named_params: Iterable[Tuple[str, torch.nn.Parameter]], dtype: torch.dtype
) -> None:
    """
    Validates that all input parameters have the expected dtype.

    Args:
        named_params (Iterable[Tuple[str, torch.nn.Parameter]]): Iterable of named parameters.
        dtype (torch.dtype): Expected dtype.

    Raises:
        ValueError: If any parameter has a different dtype than `dtype`.
    """
    for name, param in named_params:
        if param.dtype != dtype:
            raise ValueError(
                f"Parameter {name} has dtype {param.dtype}, but expected {dtype}"
            )
