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
from torchtune.training import get_world_size_and_rank

from torchtune.utils import get_logger

log = get_logger("INFO")

PROFILER_KEY = "profiler"
DEFAULT_PROFILER_ACTIVITIES = {
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
}

DEFAULT_SCHEDULE: dict = {
    "wait_steps": 5,
    "warmup_steps": 5,
    "active_steps": 2,
    "num_cycles": 1,
}

DEFAULT_TRACE_OPTS: dict = {
    "profile_memory": False,
    "with_stack": False,
    "record_shapes": True,
    "with_flops": False,
}

DEFAULT_PROFILE_DIR: str = "profiler_output"


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
        worker_name=f"r0-{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}",
        use_gzip=True,
    )
    exporter(prof)

    if rank == 0:
        log.info(f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds")

    # Memory timeline sometimes fails to export
    if prof.profile_memory:
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


class DummyProfiler:
    """
    Drop-in replacement for torch.profiler.profile that functions as a nullcontext / object
    with no-op methods for ``start``, ``stop``, and ``step``.

    This is helpful for instrumenting profiling in a recipe without requiring changes to the
    code independent of whether profiling is on / off.

    E.g.,
    ```
        profiler = DummyProfiler()
        #profiler = torch.profiler.profile()

        # Below is same regardless of profiler object type
        with profiler as prof:
            for epoch in epochs:
                for batch in batches:
                    train.step()
                    prof.step()

    """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def step(self):
        pass


def setup_torch_profiler(
    enabled: bool = False,
    cpu: bool = True,
    cuda: bool = True,
    profile_memory: bool = DEFAULT_TRACE_OPTS["profile_memory"],
    with_stack: bool = DEFAULT_TRACE_OPTS["with_stack"],
    record_shapes: bool = DEFAULT_TRACE_OPTS["record_shapes"],
    with_flops: bool = DEFAULT_TRACE_OPTS["with_flops"],
    # `torch.profiler.schedule` args - note we defer setting these to enable more fine-grained
    # warnings within this setup function
    wait_steps: Optional[int] = None,
    warmup_steps: Optional[int] = None,
    active_steps: Optional[int] = None,
    num_cycles: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Tuple[torch.profiler.profile, DictConfig]:
    """
    Sets up :class:`~torch.profiler.profile` and returns the profiler config with post-setup updates.

    The profiler config can be provided in configs under the ``profiler`` key with the following layout:

    .. code-block:: yaml

        profiler:
          _component_: torchtune.training.setup_torch_profiler
          enabled: bool
          # Output directory of trace artifacts
          output_dir: str

          # torch.profiler.ProfilerActivity types to trace
          cpu: bool
          cuda: bool

          # Trace options
          profile_memory: bool
          with_stack: bool
          record_shapes: bool
          with_flops: bool

          # torch.profiler.schedule args
          wait_steps: int
          warmup_steps: int
          active_steps: int
          num_cycles: int

    The profiler schedule updates with respect to an optimizer step (e.g., if
    ``gradient_accumulation = 2``, then the profiler will step every 2 batches).

    Sensible defaults will be chosen if the config is missing options:

    - If no activities are specified, profiler will default to CPU + CUDA
    - If no schedule is specified, profiler will default to ``DEFAULT_SCHEDULE``
    - Certain options will be overridden (``with_stack`` and ``record_shapes``) \
    depending on requirements of other options (e.g., ``profile_memory`` requires \
    ``with_stack`` and ``record_shapes``).


    Note:
        - Enabling the profiler will result in training speed reduction.
        - Setting ``profile_memory: True`` will generate large trace files.
        - The profiler schedule is context dependent. Calling ``profiler.step()`` \
        at each batch iteration but **outside** the gradient accumulation scope will \
        ``step`` the profiler each forward / backward step. Calling ``profiler.step()`` \
        each batch iteration but **within** the gradient accumulation scope  will ``step`` \
        the profiler each optimizer update step such that each ``step`` contains multiple \
        forward / backward passes.

    Args:
        enabled (bool): Enable pytorch profiler. Default is False.
        cpu (bool): Enable cpu profiling. Default is True.
        cuda (bool): Enable cuda profiling. Default is True.
        profile_memory (bool): Profile memory usage. Default is False.
        with_stack (bool): Profile stack. Default is False.
        record_shapes (bool): Record shapes. Default is True.
        with_flops (bool): Profile flops. Default is False.
        wait_steps (Optional[int]): Wait time in steps. Maps to ``wait`` kwarg of ``torch.profiler.schedule``.
        warmup_steps (Optional[int]): Warmup time in steps. Maps to ``warmup`` kwarg of ``torch.profiler.schedule``.
        active_steps (Optional[int]): Active time in steps. Maps to ``active`` kwarg of ``torch.profiler.schedule``.
        num_cycles (Optional[int]): Number of profiling cycles. Maps to ``repeat`` kwarg of ``torch.profiler.schedule``.
        output_dir (Optional[str]): Tracing file output path.

    Returns:
        Tuple[torch.profiler.profile, DictConfig]
    """

    if not enabled:
        _warn(" Profiling disabled.")
        return DummyProfiler(), DictConfig({"enabled": False})

    # Set up profiler activities
    activities = []
    if cpu:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    if len(activities) == 0:
        _warn("No activities specified, defaulting to CPU + CUDA")
        activities = DEFAULT_PROFILER_ACTIVITIES
        cpu = cuda = True

    # Check for schedule
    # 1) If no schedule is provided, set to DEFAULT_SCHEDULE
    # 2) else check for missing keys and warn if any are missing, setting these to defaults
    # Note that this might result in code duplication if these checks are already done in the `recipe`
    # However, we retain this checks in the case that the _setup_profiler section of the `recipe` does not implement these checks

    # Set up profiler schedule
    use_default_schedule = not any(
        [
            wait_steps is not None,
            warmup_steps is not None,
            active_steps is not None,
            num_cycles is not None,
        ]
    )

    # Use default schedule if None, else validate that schedule is valid and can be passed to `instantiate`
    if use_default_schedule:
        schedule_args = DEFAULT_SCHEDULE
        _warn(
            " No schedule found in config, defaulting to {}".format(
                ", ".join(f"{k} = {schedule_args[k]}" for k in schedule_args.keys())
            )
        )
    else:
        schedule_args = {
            "wait_steps": wait_steps,
            "warmup_steps": warmup_steps,
            "active_steps": active_steps,
            "num_cycles": num_cycles,
        }
        missing_keys = [k for k in schedule_args.keys() if schedule_args[k] is None]
        if len(missing_keys) > 0:
            for k in missing_keys:
                schedule_args[k] = DEFAULT_SCHEDULE[k]
            _warn(
                " Missing keys in torch profiler schedule {}: defaulting to {}".format(
                    ", ".join(missing_keys),
                    ", ".join(f"{k} = {schedule_args[k]}" for k in missing_keys),
                )
            )
    schedule = torch.profiler.schedule(
        wait=schedule_args["wait_steps"],
        warmup=schedule_args["warmup_steps"],
        active=schedule_args["active_steps"],
        repeat=schedule_args["num_cycles"],
    )

    # profile_memory requires with_stack and record_shapes, hence we override these if profile_memory is True
    # See torch.profiler.profiler._memory_profile
    if profile_memory:
        _warn(
            "`profile_memory` requires `with_stack` and `record_shapes`, these will be enabled since `profile_memory` is True"
        )
    with_stack = with_stack or profile_memory
    record_shapes = record_shapes or profile_memory
    # experimental config is needed to export stacks: see https://github.com/pytorch/pytorch/issues/100253
    experimental_config = _ExperimentalConfig(verbose=True) if with_stack else None

    # Handle exporting of trace, memory timeline and other profiler artifacts
    if output_dir is None:
        _warn(
            f" No output directory found in profiler config, defaulting to {DEFAULT_PROFILE_DIR}"
        )
        output_dir = DEFAULT_PROFILE_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)

    # trace_handler manages the export of profiler artifacts
    # this callback will be triggered after **each** profiling cycle
    callback = partial(trace_handler, output_dir=output_dir)

    profiler = torch.profiler.profile(
        activities=activities,
        profile_memory=profile_memory,
        with_stack=with_stack,
        record_shapes=record_shapes,
        with_flops=with_flops,
        schedule=schedule,
        experimental_config=experimental_config,
        on_trace_ready=callback,
    )

    profiler_cfg = DictConfig(
        {
            "enabled": enabled,
            "output_dir": output_dir,
            "cpu": cpu,
            "cuda": cuda,
            "profile_memory": profile_memory,
            "with_stack": with_stack,
            "record_shapes": record_shapes,
            "with_flops": with_flops,
            **schedule_args,
        }
    )

    return (profiler, profiler_cfg)
