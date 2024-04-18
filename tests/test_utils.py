# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import re
import sys
import unittest
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Generator, Optional, TextIO, Tuple, Union

import pytest

import torch
from torch import nn
from torchtune.modules.tokenizers import SentencePieceTokenizer

skip_if_cuda_not_available = unittest.skipIf(
    not torch.cuda.is_available(), "CUDA is not available"
)

CKPT_MODEL_PATHS = {
    "small_test_ckpt_tune": "/tmp/test-artifacts/small-ckpt-tune-03082024.pt",
    "small_test_ckpt_meta": "/tmp/test-artifacts/small-ckpt-meta-03082024.pt",
    "small_test_ckpt_hf": "/tmp/test-artifacts/small-ckpt-hf-03082024.pt",
    "llama2_7b": "/tmp/test-artifacts/llama2-7b-torchtune.pt",
}


def torch_version_ge(version: str) -> bool:
    """
    Check if torch version is greater than or equal to the given version
    """
    return version in torch.__version__ or torch.__version__ >= version


# Inherit from SentencePieceTokenizer class to reuse its tokenize_messages method
class DummyTokenizer(SentencePieceTokenizer):
    def __init__(self):
        self.encodes_whitespace = False

    def encode(self, text, add_bos=True, add_eos=True, **kwargs):
        words = text.split()
        tokens = [len(word) for word in words]
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    @property
    def eos_id(self):
        return -1

    @property
    def bos_id(self):
        return 0


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
    dtype: Optional[torch.dtype] = None,
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
                dtype=param.dtype if dtype is None else dtype,
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


def gpu_test(gpu_count: int = 1):
    """
    Annotation for GPU tests, skipping the test if the
    required amount of GPU is not available
    """
    message = f"Not enough GPUs to run the test: requires {gpu_count}"
    local_gpu_count: int = torch.cuda.device_count()
    return pytest.mark.skipif(local_gpu_count < gpu_count, reason=message)


def get_loss_values_from_metric_logger(log_file_path: str) -> Dict[str, float]:
    """
    Given an output directory containing metric logger .txt file,
    parse the .txt and return a list of losses from each logged iteration.
    """
    with open(log_file_path, "r") as f:
        logs = f.read()
    losses = [float(x) for x in re.findall(r"loss:(\d+\.\d+)", logs)]
    return losses


def gen_log_file_name(tmpdir, suffix: Optional[str] = None) -> str:
    """
    Take the tmpdir and just append a non-path version of it as the
    filename, optionally adding specified suffix. This is used to
    write metric logs to a deterministic file per test run.
    E.g. /tmp/my/dir -> /tmp/my/dir/tmpmydir.txt
    """
    filename = str(tmpdir) + str(tmpdir).replace("/", "")
    if suffix:
        filename += suffix
    filename += ".txt"
    return filename
