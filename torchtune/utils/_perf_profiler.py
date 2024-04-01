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
        output_dir (Optional[str]): Tracing file output path.

    Returns:
        ContextManager: pytorch profiler context manager
    """

    def trace_handler(prof) -> None:
        try:
            prof.export_chrome_trace(output_dir)
        except Exception as e:
            raise Exception(
                f"torch profiler failed to output trace to {output_dir}"
            ) from e

    return profile(
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
