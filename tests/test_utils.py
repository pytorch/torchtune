# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import sys
import unittest
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, Generator, TextIO, Tuple, Union

import torch
from torch import nn


skip_if_cuda_not_available = unittest.skipIf(
    not torch.cuda.is_available(), "CUDA is not available"
)


def get_assets_path():
    return Path(__file__).parent / "assets"


def init_weights_with_constant(model: nn.Module, constant: float = 1.0) -> None:
    for p in model.parameters():
        nn.init.constant_(p, constant)


def fixed_init_tensor(
    shape: torch.Size,
    min_val: Union[float, int] = 0.0,
    max_val: Union[float, int] = 1.0,
    nonlinear: bool = False,
    dtype: torch.dtype = torch.float,
):
    """
    Utility for generating deterministic tensors of a given shape. In general stuff
    like torch.ones, torch.eye, etc can result in trivial outputs. This utility
    generates a range tensor [min_val, max_val) of a specified dtype, applies
    a sine function if nonlinear=True, then reshapes to the appropriate shape.
    """
    n_elements = math.prod(shape)
    step_size = (max_val - min_val) / n_elements
    x = torch.arange(min_val, max_val, step_size, dtype=dtype)
    x = x.reshape(shape)
    if nonlinear:
        return torch.sin(x)
    return x


@torch.no_grad
def fixed_init_model(
    model: nn.Module,
    min_val: Union[float, int] = 0.0,
    max_val: Union[float, int] = 1.0,
    nonlinear: bool = False,
):
    """
    This utility initializes all parameters of a model deterministically using the
    function fixed_init_tensor above. See that docstring for details of each parameter.
    """
    for _, param in model.named_parameters():
        param.copy_(
            fixed_init_tensor(
                param.shape,
                min_val=min_val,
                max_val=max_val,
                nonlinear=nonlinear,
                dtype=param.dtype,
            )
        )


def assert_expected(
    actual: Any,
    expected: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_device: bool = True,
):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        check_device=check_device,
        msg=f"actual: {actual}, expected: {expected}",
    )


@contextmanager
def single_box_init(init_pg: bool = True):
    env_vars = ["MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK", "RANK", "WORLD_SIZE"]
    initial_os = {k: os.environ.get(k, None) for k in env_vars}
    os.environ.get("MASTER_ADDR", None)
    os.environ["MASTER_ADDR"] = "localhost"
    # TODO: Don't hardcode ports as this could cause flakiness if tests execute
    # in parallel.
    os.environ["MASTER_PORT"] = str(12345)
    os.environ["LOCAL_RANK"] = str(0)
    os.environ["RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(1)
    if init_pg:
        torch.distributed.init_process_group(
            backend="gloo",
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"]),
        )
    try:
        yield
    finally:
        if init_pg:
            torch.distributed.destroy_process_group()
        for k in env_vars:
            if initial_os.get(k) is None:
                del os.environ[k]
            else:
                os.environ[k] = initial_os[k]


@contextmanager
def set_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


@contextmanager
def captured_output() -> Generator[Tuple[TextIO, TextIO], None, None]:
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err
