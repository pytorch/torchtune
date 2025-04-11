# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import datetime
import os
import time
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed

from omegaconf import DictConfig
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import tensorboard_trace_handler

from torchtune.utils import get_logger, get_world_size_and_rank

log = get_logger("INFO")

PROFILER_KEY = "profiler"


def _warn(msg: str):
    _, rank = get_world_size_and_rank()
    if rank == 0:
        log.warning(msg)


def trace_handler(
    prof: torch.profiler.profile,
    output_dir: str,
    metric: str = "self_cuda_time_total",
    row_limit: int = 25,
):
    """
    Handles export of artifacts from ``torch.profiler.profile``.

    The following artifacts are exported:
    - chrome / tensorboard trace - viewable through tensorboard or perfetto.dev / chrome::/tracing
    - trace event table
    - memory timeline and snapshot.pickle if ``profile_memory``
    - stacks if ``with_stack`` (note that ``profile_memory`` requires ``with_stack`` to be ``True``),
    viewable as a flamegraph see (https://pytorch.org/docs/stable/profiler.html#torch.profiler._KinetoProfile.export_stacks).

    Notes:
    - Each profiling cycle is exported as a sub-directory in output_dir
        - E.g., profiling in 5-step cycle (wait=2, warmup=2, active=1, repeat=0) will result in
        sub-directories iteration_5, iteration_10, etc.
    - If profiling in a distributed setting, each artifact will be prefixed with rank.
    - Memory timeline is only exported for rank 0 (error if exporting from multiple ranks on single node)

    See profiler documentation (https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile) for more details

    Args:
        prof (torch.profiler.profile): instance of torch profiler to use
        output_dir (str):  directory to store artifacts
        metric (str): metric to order trace event table by, see ``torch.profiler.profile.key_averages().table`` for
        row_limit (int): number of rows to display in trace event table

    """
    world_size, rank = get_world_size_and_rank()
    curr_trace_dir_name = "iteration_" + str(prof.step_num)
    curr_trace_dir = os.path.join(output_dir, curr_trace_dir_name)
    if not os.path.exists(curr_trace_dir):
        os.makedirs(curr_trace_dir, exist_ok=True)

    # Export chrome / tensorboard trace
    if rank == 0:
        log.info(f"Dumping traces at step {prof.step_num}")
    begin = time.monotonic()

    # Use tensorboard trace handler rather than directly exporting chrome traces since
    # tensorboard doesn't seem to be able to parse traces with prof.export_chrome_trace

    now = datetime.datetime.now()

    exporter = tensorboard_trace_handler(
        curr_trace_dir,
        worker_name=f"r{rank}-{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}",
        use_gzip=True,
    )
    exporter(prof)

    if rank == 0:
        log.info(f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds")

    # Memory timeline sometimes fails to export
    if prof.profile_memory and torch.cuda.is_available():
        if rank == 0:
            try:
                prof.export_memory_timeline(
                    f"{curr_trace_dir}/rank{rank}_memory-timeline.html"
                )
            except Exception as e:
                log.warn(f" Failed to export memory timeline: {e}")

            torch.cuda.memory._dump_snapshot(
                f"{curr_trace_dir}/rank{rank}_memory_snapshot.pickle"
            )

    # Dump stack traces
    if prof.with_stack:
        prof.export_stacks(f"{curr_trace_dir}/rank{rank}_stacks.txt", metric=metric)

    # Export event averages
    key_avgs = prof.key_averages(
        group_by_input_shape=prof.record_shapes, group_by_stack_n=5
    ).table(sort_by=metric, row_limit=row_limit)
    with open(f"{curr_trace_dir}/rank{rank}_key_averages.txt", "w") as f:
        print(key_avgs, file=f)
    if rank == 0:
        log.info(f"Saving profiling results to {curr_trace_dir}")

    # TODO: Is this necessary?
    # see https://github.com/pytorch/torchtitan/blob/3050098dcee4901d88c712f9e8e9703d1735a29b/torchtitan/profiling.py#L48
    if world_size > 1:
        torch.distributed.barrier()


from typing import Protocol


class Profiler(Protocol):
    """Protocol for a memory profiler in torchtune.

    Example:
    ```python
        with Profiler() as profiler:
            for iter in range(10):
                # code to profile
                profiler.step()
    ```
    """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    # def start(self):
    #     pass

    # def stop(self):
    #     pass

    def step(self):
        pass

    # def setup_torch_profiler(
    #     enabled: bool = False,
    #     cpu: bool = True,
    #     cuda: bool = True,
    #     xpu: bool = True,
    #     profile_memory: bool = DEFAULT_TRACE_OPTS["profile_memory"],
    #     with_stack: bool = DEFAULT_TRACE_OPTS["with_stack"],
    #     record_shapes: bool = DEFAULT_TRACE_OPTS["record_shapes"],
    #     with_flops: bool = DEFAULT_TRACE_OPTS["with_flops"],
    #     # `torch.profiler.schedule` args - note we defer setting these to enable more fine-grained
    #     # warnings within this setup function
    #     wait_steps: Optional[int] = None,
    #     warmup_steps: Optional[int] = None,
    #     active_steps: Optional[int] = None,
    #     num_cycles: Optional[int] = None,
    #     output_dir: Optional[str] = None,
    # ) -> Tuple[torch.profiler.profile, DictConfig]:
    #     """
    #     Sets up :class:`~torch.profiler.profile` and returns the profiler config with post-setup updates.

    #     The profiler config can be provided in configs under the ``profiler`` key with the following layout:

    #     .. code-block:: yaml

    #         profiler:
    #           _component_: torchtune.training.setup_torch_profiler
    #           enabled: bool
    #           # Output directory of trace artifacts
    #           output_dir: str

    #           # torch.profiler.ProfilerActivity types to trace
    #           cpu: bool
    #           cuda: bool

    #           # Trace options
    #           profile_memory: bool
    #           with_stack: bool
    #           record_shapes: bool
    #           with_flops: bool

    #           # torch.profiler.schedule args
    #           wait_steps: int
    #           warmup_steps: int
    #           active_steps: int
    #           num_cycles: int

    #     The profiler schedule updates with respect to an optimizer step (e.g., if
    #     ``gradient_accumulation = 2``, then the profiler will step every 2 batches).

    #     Sensible defaults will be chosen if the config is missing options:

    #     - If no activities are specified, profiler will default to CPU + CUDA
    #     - If no schedule is specified, profiler will default to ``DEFAULT_SCHEDULE``
    #     - Certain options will be overridden (``with_stack`` and ``record_shapes``) \
    #     depending on requirements of other options (e.g., ``profile_memory`` requires \
    #     ``with_stack`` and ``record_shapes``).

    #     Note:
    #         - Enabling the profiler will result in training speed reduction.
    #         - Setting ``profile_memory: True`` will generate large trace files.
    #         - The profiler schedule is context dependent. Calling ``profiler.step()`` \
    #         at each batch iteration but **outside** the gradient accumulation scope will \
    #         ``step`` the profiler each forward / backward step. Calling ``profiler.step()`` \
    #         each batch iteration but **within** the gradient accumulation scope  will ``step`` \
    #         the profiler each optimizer update step such that each ``step`` contains multiple \
    #         forward / backward passes.

    #     Args:
    #         enabled (bool): Enable pytorch profiler. Default is False.
    #         cpu (bool): Enable cpu profiling. Default is True.
    #         cuda (bool): Enable cuda profiling. Default is True.
    #         xpu (bool): Enable xpu profiling. Default is True.
    #         profile_memory (bool): Profile memory usage. Default is False.
    #         with_stack (bool): Profile stack. Default is False.
    #         record_shapes (bool): Record shapes. Default is True.
    #         with_flops (bool): Profile flops. Default is False.
    #         wait_steps (Optional[int]): Wait time in steps. Maps to ``wait`` kwarg of ``torch.profiler.schedule``.
    #         warmup_steps (Optional[int]): Warmup time in steps. Maps to ``warmup`` kwarg of ``torch.profiler.schedule``.
    #         active_steps (Optional[int]): Active time in steps. Maps to ``active`` kwarg of ``torch.profiler.schedule``.
    #         num_cycles (Optional[int]): Number of profiling cycles. Maps to ``repeat`` kwarg of ``torch.profiler.schedule``.
    #         output_dir (Optional[str]): Tracing file output path.

    #     Returns:
    #         Tuple[torch.profiler.profile, DictConfig]
    #     """
    # pass


class TorchProfiler(Profiler, torch.profiler.profile):

    def __init__(
        self,
        *,
        cuda: bool = False,
        cpu: bool = False,
        xpu: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        record_shapes: bool = True,
        with_flops: bool = False,
        wait_steps: int = 5,
        warmup_steps: int = 3,
        active_steps: int = 2,
        num_cycles: int = 1,
        output_dir: str = "profiler_output",
    ):
        self.cuda = cuda
        self.cpu = cpu
        self.xpu = xpu
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.wait_steps = wait_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.num_cycles = num_cycles
        self.output_dir = output_dir

        # Collect all activities to profile
        activities = []
        if self.cpu:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if self.cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if self.xpu:
            activities.append(torch.profiler.ProfilerActivity.XPU)
        self.enabled = len(activities) > 0

        # Init the schedule
        schedule = torch.profiler.schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=active_steps,
            repeat=num_cycles,
        )

        # profile_memory requires with_stack and record_shapes, hence we override these if profile_memory is True
        # See torch.profiler.profiler._memory_profile
        if profile_memory:
            log.DEBUG(
                "`profile_memory` requires `with_stack` and `record_shapes`, these will be enabled since `profile_memory` is True"
            )
        with_stack = with_stack or profile_memory
        record_shapes = record_shapes or profile_memory
        # experimental config is needed to export stacks: see https://github.com/pytorch/pytorch/issues/100253
        experimental_config = _ExperimentalConfig(verbose=True) if with_stack else None

        # Create output dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # trace_handler manages the export of profiler artifacts
        # this callback will be triggered after **each** profiling cycle
        callback = partial(trace_handler, output_dir=self.output_dir)

        # Finally, create the profiler
        if self.enabled:
            super().__init__(
                activities=activities,
                profile_memory=profile_memory,
                with_stack=with_stack,
                record_shapes=record_shapes,
                with_flops=with_flops,
                schedule=schedule,
                experimental_config=experimental_config,
                on_trace_ready=callback,
            )
            self.curr_step = 0
            log.info(f"Profiler writing to: {output_dir.resolve()}")
        else:
            log.info("Profiler is disabled")
            self.curr_step = -1
        self.start_time = time.perf_counter()
        self.cuda_memory_step = self.wait_steps + self.warmup_steps
        # Start recording on init to capture first step
        if self.curr_step == self.cuda_memory_step:
            self._record_memory_history()

    def _record_memory_history(self):
        if self.rank == 0 and self.device.type == "cuda":
            torch.cuda.memory._record_memory_history()

    def _end_record_memory_history(self):
        if self.rank == 0 and self.device.type == "cuda":
            torch.cuda.memory._record_memory_history(enabled=None)

    def step(self):
        if self.enabled:
            super().step()
        self.start_time = time.perf_counter()
        self.curr_step += 1
        if self.curr_step == self.cuda_memory_step:
            torch.cuda.memory._record_memory_history()
        elif self.curr_step == self.cuda_memory_step + self.active_steps:
            torch.cuda.memory._record_memory_history(enabled=None)

    @property
    def step_time(self):
        return time.perf_counter() - self.start_time

    # def __enter__(self):
    #     if self.enabled:
    #         self.profiler.start()
    #     return self

    # def __exit__(self, type, value, traceback):
    #     if self.enabled:
    #         self.profiler.stop()
