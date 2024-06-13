# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

import os
import time
from functools import partial
from pathlib import Path
from typing import ContextManager, Optional, Tuple

import torch
import torch.distributed

from omegaconf import DictConfig
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import profile, tensorboard_trace_handler
from torchtune.utils import get_world_size_and_rank

from torchtune.utils.logging import get_logger

log = get_logger("INFO")


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


PROFILER_KEY = "profiler"
_DEFAULT_PROFILER_ACTIVITIES = {
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
}

_DEFAULT_SCHEDULE: dict = {
    "wait": 5,
    "warmup": 5,
    "active": 2,
    "repeat": 1,
}

DEFAULT_TRACE_OPTS: dict = {
    "profile_memory": False,
    "with_stack": False,
    "record_shapes": True,
    "with_flops": False,
}

_DEFAULT_PROFILE_DIR: str = "profiler_output"


def _warn(msg: str):
    _, rank = get_world_size_and_rank()
    if rank == 0:
        log.warn(msg)


def trace_handler(
    prof: torch.profiler.profiler.profile,
    output_dir,
    metric="self_cuda_time_total",
    row_limit=25,
):
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
    exporter = tensorboard_trace_handler(
        curr_trace_dir, worker_name=f"rank{rank}", use_gzip=True
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


class FakeProfiler:
    """
    Drop-in replacement for torch.profiler.profile that functions as a nullcontext / object
    with no-op methods for `start`, `stop`, and `step`.

    This is helpful for instrumenting profiling in a recipe without requiring changes to the
    code independent of whether profiling is on / off.

    E.g.,
    ```
        profiler = FakeProfiler()
        #profiler = torch.profiler.profile()

        # Below is same regardless of profiler object type
        with profiler as prof:
            for epoch in epochs:
                for batch in batches:
                    train.step()
                    prof.step()
    ```
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


def should_profile(cfg: DictConfig) -> bool:
    return (cfg is not None and len(cfg) > 0) and cfg.get("enabled", True)


# NOTE: This is a reference implementation of a profiler setup method to be defined within a `recipe`.
# Its purpose is to parse high-level configurations and pass the parsed args to the actual profiler setup.
# It is not exported as the user-facing profiler setup functions (currently only `setup_torch_profiler`)
# should be used instead.
def _setup_profiler(cfg: DictConfig, log_cfg: bool = False) -> torch.profiler.profile:
    """
    Parses the `profiler` section of top-level `cfg` and sets up profiler

    Args:
        cfg: DictConfig - `profiler` section of the top-level `cfg` (the main config passed to `recipe.main`)
        log_cfg: bool - whether to return the profiler config after profiler setup, which sets defaults and possibly
        overrides certain profiling options.

        NOTE: Since not all settings of the profiler can be parsed from the returned profiler object,
        such as the `schedule`, `log_cfg` can be used for easy logging / debugging of all profiler options post setup.

    Returns:
        profiler: torch.profiler.profile | FakeProfiler - FakeProfiler is a nullcontext with no-op methods
        for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
        that the instrumented training loop does not need to be changed profiling is disabled.
        profiler_cfg: Optional[DictConfig]

    The profiler config can be provided in configs under the `profiler` key with the following layout:
    ```
    profiler:
        enabled: bool

        #Output directory of trace artifacts
        output_dir: str

        #`torch.profiler.ProfilerActivity` types to trace
        CPU: bool
        CUDA: bool

        #Trace options
        profile_memory: bool
        with_stack: bool
        record_shapes: bool
        with_flops: bool

        #`torch.profiler.schedule` args
        schedule:
            wait: int
            warmup: int
            active: int
            repeat: int
    ```
    """
    if should_profile(cfg):
        enabled = True
    else:
        return FakeProfiler()

    # Set up profiler activities
    cpu = cfg.get("CPU", False)
    cuda = cfg.get("CUDA", False)
    profile_memory = cfg.get("profile_memory", DEFAULT_TRACE_OPTS["profile_memory"])
    with_stack = cfg.get("with_stack", DEFAULT_TRACE_OPTS["with_stack"])
    record_shapes = cfg.get("record_shapes", DEFAULT_TRACE_OPTS["record_shapes"])
    with_flops = cfg.get("with_flops", DEFAULT_TRACE_OPTS["with_flops"])
    output_dir = cfg.get("output_dir", None)

    # Parse schedule specific args
    schedule_cfg = cfg.get("schedule", None)

    if schedule_cfg is None:
        wait = None
        warmup = None
        active = None
        repeat = None
    else:
        wait = schedule_cfg.get("wait", None)
        warmup = schedule_cfg.get("warmup", None)
        active = schedule_cfg.get("active", None)
        repeat = schedule_cfg.get("repeat", None)
    # Delegate setup of actual profiler and optionally return updated profiler config
    profiler = setup_torch_profiler(
        enabled=enabled,
        cpu=cpu,
        cuda=cuda,
        profile_memory=profile_memory,
        with_stack=with_stack,
        record_shapes=record_shapes,
        with_flops=with_flops,
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
        output_dir=output_dir,
        return_cfg=log_cfg,
    )
    if log_cfg:
        profiler, profiler_cfg = profiler
        log.info(f" Profiler config after instantiation: {profiler_cfg}")

    return profiler


def setup_torch_profiler(
    enabled: bool = False,
    cpu: bool = True,
    cuda: bool = True,
    profile_memory: bool = False,
    with_stack: bool = False,
    record_shapes: bool = False,
    with_flops: bool = False,
    wait: Optional[int] = None,
    warmup: Optional[int] = None,
    active: Optional[int] = None,
    repeat: Optional[int] = None,
    output_dir: Optional[str] = None,
    return_cfg: bool = False,
) -> Tuple[torch.profiler.profile, Optional[DictConfig]]:
    """
    Sets up torch.profiler.profile and optionally returns profiler config

    The profiler config can be provided in configs under the `profiler` key with the following layout:
    ```
    profiler:
        enabled: bool

        #Output directory of trace artifacts
        output_dir: str

        #`torch.profiler.ProfilerActivity` types to trace
        CPU: bool
        CUDA: bool

        #Trace options
        profile_memory: bool
        with_stack: bool
        record_shapes: bool
        with_flops: bool

        #`torch.profiler.schedule` args
        schedule:
            wait: int
            warmup: int
            active: int
            repeat: int
    ```

    Args:
        enabled (bool): Enable pytorch profiler. Default is False.
        cpu (bool): Enable cpu profiling. Default is True.
        cuda (bool): Enable cuda profiling. Default is True.
        profile_memory (bool): Profile memory usage. Default is False.
        with_stack (bool): Profile stack. Default is False.
        record_shapes (bool): Record shapes. Default is False.
        with_flops (bool): Profile flops. Default is False.
        wait (Optional[int]): Wait time in steps. Default is None.
        warmup (Optional[int]): Warmup time in steps. Default is None.
        active (Optional[int]): Active time in steps. Default is None.
        repeat (Optional[int]): Repeat time in steps. Default is None.
        output_dir (Optional[str]): Tracing file output path. Default is None.
        return_cfg (bool): Return updated profiler config. Default is False.

    Returns:
        tuple: [torch.profiler.profile, Optional[DictConfig]]

    NOTE:
    - Enabling the profiler will result in training speed reduction.
    - Setting `profile_memory: true` will generate large trace files.
    - The profiler schedule is context dependent:
        - Calling `profiler.step()` at each batch iteration but outside the gradient accumulation
        scope will `step` the profiler each forward / backward step
        - Calling `profiler.step()` each batch iteration but within the gradient accumulation scope
        will `step` the profiler each optimizer update step such that each `step` contains multiple
        forward / backward passes.

    Additional notes:
        - the profiler schedule updates with respect to an optimizer step:
            - e.g., if `gradient_accumulation = 2`, then the profiler will step every 2 batches.
        - sensible defaults will be chosen if the config is missing options
            - if no activities are specified, profiler will default to CPU + CUDA
            - if no schedule is specified, profiler will default to wait 10, warmup 5, active 3, repeat 1
            - if a schedule is specified, profiler will validate that the schedule is valid and can be passed to `instantiate`
            - certain options will be overridden (`with_stack` and `record_shapes`) depending on requirements of other options
                - e.g., `profile_memory` requires `with_stack` and `record_shapes`
    """

    # Set up profiler activities
    activities = []
    if cpu:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    if len(activities) == 0:
        _warn("No activities specified, defaulting to CPU + CUDA")
        activities = _DEFAULT_PROFILER_ACTIVITIES

    # Set up profiler schedule
    use_default_schedule = not any([wait, warmup, active, repeat])

    # Use default schedule if None, else validate that schedule is valid and can be passed to `instantiate`
    if use_default_schedule:
        wait = _DEFAULT_SCHEDULE["wait"]
        warmup = _DEFAULT_SCHEDULE["warmup"]
        active = _DEFAULT_SCHEDULE["active"]
        repeat = _DEFAULT_SCHEDULE["repeat"]
        _warn(
            " No schedule found in config, defaulting to wait {wait}, warmup {warmup}, active {active}, repeat {repeat}"
        )
    else:
        if not all([wait, warmup, active]):
            raise ValueError(
                "Invalid schedule config: must specify wait, warmup, and active"
            )
        if repeat is None:
            _warn(
                """ No repeat found in schedule config, setting to 1 (one cycle).
                If you want to cycle continuously, specify `repeat = 0`"""
            )
            repeat = 1

    schedule = torch.profiler.schedule(
        wait=wait, warmup=warmup, active=active, repeat=repeat
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
            f" No output directory found in profiler config, defaulting to {_DEFAULT_PROFILE_DIR}"
        )
        output_dir = _DEFAULT_PROFILE_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)
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
    if return_cfg:
        profiler_cfg = DictConfig(
            {
                "enabled": enabled,
                "output_dir": output_dir,
                "CPU": cpu,
                "CUDA": cuda,
                "profile_memory": profile_memory,
                "with_stack": with_stack,
                "record_shapes": record_shapes,
                "with_flops": with_flops,
                "schedule": {
                    "wait": wait,
                    "warmup": warmup,
                    "active": active,
                    "repeat": repeat,
                },
            }
        )
        return (profiler, profiler_cfg)
    return profiler
