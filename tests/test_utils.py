# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Union

import numpy as np

import torch

from torch import nn


def set_rng_seed(seed):
    """Sets the seed for random number generators"""
    torch.manual_seed(seed)
    random.seed(seed)


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
    n_elements = np.prod(shape)
    step_size = (max_val - min_val) / n_elements
    print(f"RV: step_size {step_size}")
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
