# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import ContextManager, Optional

import torch
from torch.profiler import profile


def perf_profiler(
    output_dir: Optional[str] = "./torchtune_perf_tracing.json",
) -> ContextManager:
    """
    Utility component to trace the code with pytorch profiler.
    check the user manual: https://pytorch.org/docs/stable/profiler.html for more details.
    The schedule for this profiler is wait 5 steps, warmup 5 steps, trace 5 steps
    Note: Enable pytorch profiler may casue performance overhead.

    Args:
        enabled (Optional[bool]): Whether enable pytorch profiler or not.
        output_file_path (Optional[str]): Tracing file output path.
        is_rank_zero (Optional[bool]): Whether the current rank is zero.

    Returns:
        ContextManager: pytorch profiler context manager
    """

    def trace_handler(prof) -> None:
        prof.export_chrome_trace(output_file_path)

    return profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=5, repeat=1),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
