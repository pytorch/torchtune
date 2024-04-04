# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

from typing import ContextManager, Optional

import torch
from torch.profiler import profile


def profiler(
    enabled: Optional[bool] = False,
    output_dir: Optional[str] = "./torchtune_perf_tracing.json",
) -> ContextManager:
    """
    Utility component that wraps around `torch.profiler` to profile model's operators.
    See https://pytorch.org/docs/stable/profiler.html for more details.
    The schedule for this profiler is wait 100 steps, warmup 5 steps, trace 5 steps
    Note: Enabling pytorch profiler may have training speed reduction.

    Args:
        enabled (Optional[bool]): Enable pytorch profiler. Default is False.
        output_dir (Optional[str]): Tracing file output path. Default is "./torchtune_perf_tracing.json".

    Returns:
        ContextManager: pytorch profiler context manager
    """

    def trace_handler(prof) -> None:
        prof.export_chrome_trace(output_dir)

    return (
        profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=100, warmup=5, active=5, repeat=1),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        )
        if enabled
        else contextlib.nullcontext()
    )
