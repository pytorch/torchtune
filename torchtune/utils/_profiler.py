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
from typing import ContextManager, Optional

import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import profile, tensorboard_trace_handler
from torchtune import config
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
    "_component_": "torch.profiler.schedule",
    "wait": 5,
    "warmup": 5,
    "active": 2,
    "repeat": 1,
}

_DEFAULT_SCHEDULE_SINGLE: dict = {
    "_component_": "torch.profiler.schedule",
    "wait": 100,
    "warmup": 5,
    "active": 5,
    "repeat": 1,
}

_DEFAULT_SCHEDULE_DISTRIBUTED: dict = {
    "_component_": "torch.profiler.schedule",
    "wait": 5,
    "warmup": 5,
    "active": 1,
    "repeat": 1,
}
_DEFAULT_PROFILER_OPTS: dict = {
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
    row_limit=-1,
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
    Mock object that minimally mimics behavior of torch.profiler.profile

    Essentially a nullcontext with no-op methods for `start`, `stop`, and `step`
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
    return cfg.get(PROFILER_KEY, None) is not None and cfg[PROFILER_KEY].get(
        "enabled", True
    )

def parse_profiler_cfg(cfg: DictConfig) -> torch.profiler.profile:
    enabled = True
    if not should_profile(cfg):
        enabled = False

    # Set up profiler activities
    cpu = cfg[PROFILER_KEY].get("CPU", False)
    cuda = cfg[PROFILER_KEY].get("CUDA", False)
    profile_memory = cfg[PROFILER_KEY].get("profile_memory", False)
    with_stack = cfg[PROFILER_KEY].get("with_stack", False)
    record_shapes = cfg[PROFILER_KEY].get("record_shapes", False)
    with_flops = cfg[PROFILER_KEY].get("with_flops", False)
    output_dir = cfg[PROFILER_KEY].get("output_dir", None)    
    
    # Parse schedule specific args
    schedule_cfg = cfg[PROFILER_KEY].get("schedule", None)

    if schedule_cfg is None:
        wait = None
        warmup = None
        active = None
        repeat = None
    else:
        if not all(k in schedule_cfg for k in ["wait", "warmup", "active"]):
            raise ValueError(
                "Invalid schedule config: must specify wait, warmup, and active"
            )
        if "repeat" not in schedule_cfg:
            _warn(
                """ No repeat found in schedule config, setting to 1 (one cycle).
                If you want to cycle continuously, specify repeat = 0"""
            )
            repeat = 1

    profiler = setup_torch_profiler(enabled=enabled,
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
                                    output_dir=output_dir
                                    )
    return profiler

def setup_torch_profiler(enabled: bool = False,
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
                         ) -> torch.profiler.profile:
    """
    Sets up torch.profiler.profile

    NOTE: 
    - Enabling the profiler may have training speed reduction.
    - Setting `profile_memory: true` will result in large trace files.
    - The profiler schedule is context dependent:
        - Calling `profiler.step()` at each batch iteration but outside the gradient accumulation scope will `step` the profiler each forward / backward step
        - Calling `profiler.step()` each batch iteration but within the gradient accumulation scope will `step` the profiler each optimizer update step such that
        each `step` contains multiple forward / backward passes.

    Args:
        cfg (DictConfig): profiler config with following options:
        ```
        profiler:
            enabled: bool

            #Output directory of trace artifacts
            output_dir: str

            #`torch.profiler.ProfilerActivity` types to trace
            CPU: bool
            CUDA: bool
            
            #trace options
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
    Returns:
        torch.profiler.profile | FakeProfiler

    Additional notes:
        - `cfg` is modified in-place with the defaults per the comments below
        - the profiler schedule updates with respect to an optimizer step:
            - e.g., if `gradient_accumulation = 2`, then the profiler will step every 2 batches.
        - sensible defaults will be chosen if the config is missing options
            - if no activities are specified, profiler will default to CPU + CUDA
            - if no schedule is specified, profiler will default to wait 10, warmup 5, active 3, repeat 1
            - if a schedule is specified, profiler will validate that the schedule is valid and can be passed to `instantiate`
            - certain options will be overridden (`with_stack` and `record_shapes`) depending on requirements of other options
                - e.g., `profile_memory` requires `with_stack` and `record_shapes`
        - if no profiler config is found or the `cfg.enabled=False`, a fake profiler will be returned that
        minimally mimicks the interface of torch.profiler.profile (context decorator with `step` method)
    """
    if not enabled:
        return FakeProfiler()

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
        _warn(" No schedule found in config, defaulting to wait {wait}, warmup {warmup}, active {active}, repeat {repeat}")
    else:
        if not all([wait, warmup, active]):
            raise ValueError(
                "Invalid schedule config: must specify wait, warmup, and active"
            )
        if repeat is None:
            _warn(
                """ No repeat found in schedule config, setting to 1 (one cycle).
                If you want to cycle continuously, specify repeat = 0"""
            )
            repeat = 1
    
    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

    # profile_memory requires with_stack and record_shapes, hence we override these if profile_memory is True
    # See torch.profiler.profiler._memory_profile
    if profile_memory:
        _warn("`profile_memory` requires `with_stack` and `record_shapes`, these will be enabled since `profile_memory` is True")
    with_stack = (
        with_stack
        or profile_memory
    )
    record_shapes = (
        record_shapes
        or profile_memory
    )
    # experimental config is needed to export stacks: see https://github.com/pytorch/pytorch/issues/100253
    experimental_config = _ExperimentalConfig(verbose=True) if with_stack else None

    # Handle exporting of trace, memory timeline and other profiler artifacts
    if output_dir is None:
        _warn(
            f" No output directory found in profiler config, defaulting to {_DEFAULT_PROFILE_DIR}"
        )
        profiler_output_dir = _DEFAULT_PROFILE_DIR

    output_dir = Path(profiler_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    return profiler

def _setup_torch_profiler(cfg: DictConfig) -> torch.profiler.profile:
    """
    Sets up torch.profiler.profile

    NOTE: 
    - Enabling the profiler may have training speed reduction.
    - Setting `profile_memory: true` will result in large trace files.
    - The profiler schedule is context dependent:
        - Calling `profiler.step()` at each batch iteration but outside the gradient accumulation scope will `step` the profiler each forward / backward step
        - Calling `profiler.step()` each batch iteration but within the gradient accumulation scope will `step` the profiler each optimizer update step such that
        each `step` contains multiple forward / backward passes.

    Args:
        cfg (DictConfig): profiler config with following options:
        ```
        profiler:
            enabled: bool

            #Output directory of trace artifacts
            output_dir: str

            #`torch.profiler.ProfilerActivity` types to trace
            CPU: bool
            CUDA: bool

            #`torch.profiler.profile` options
            profile:
                # _component_ is optional as the setup method will handle
                _component_: torch.profiler.profile
                profile_memory: bool
                with_stack: bool
                record_shapes: bool
                with_flops: bool

            #`torch.profiler.schedule` options
            schedule:
                # _component_ is optional as the setup method will handle
                _component_: torch.profiler.schedule
                wait: int
                warmup: int
                active: int
                repeat: int
            ```
    Returns:
        torch.profiler.profile | FakeProfiler

    Additional notes:
        - `cfg` is modified in-place with the defaults per the comments below
        - the profiler schedule updates with respect to an optimizer step:
            - e.g., if `gradient_accumulation = 2`, then the profiler will step every 2 batches.
        - sensible defaults will be chosen if the config is missing options
            - if no activities are specified, profiler will default to CPU + CUDA
            - if no schedule is specified, profiler will default to wait 10, warmup 5, active 3, repeat 1
            - if a schedule is specified, profiler will validate that the schedule is valid and can be passed to `instantiate`
            - certain options will be overridden (`with_stack` and `record_shapes`) depending on requirements of other options
                - e.g., `profile_memory` requires `with_stack` and `record_shapes`
        - if no profiler config is found or the `cfg.enabled=False`, a fake profiler will be returned that
        minimally mimicks the interface of torch.profiler.profile (context decorator with `step` method)
    """

    if not should_profile(cfg):
        OmegaConf.update(cfg, f"{PROFILER_KEY}.enabled", False)
        return FakeProfiler()

    cfg[PROFILER_KEY].enabled = cfg[PROFILER_KEY].get("enabled", True)
    torch_profiler_cfg = cfg[PROFILER_KEY].get("profile", None)
    if torch_profiler_cfg is None:
        _warn(
            f" Missing torch profiler config, instantiating with default settings: {_DEFAULT_PROFILER_OPTS}"
        )
        cfg[PROFILER_KEY].profile = torch_profiler_cfg = OmegaConf.create(
            _DEFAULT_PROFILER_OPTS
        )

    # Set up profiler activities
    activities = []
    profile_cpu = cfg[PROFILER_KEY].get("CPU", False)
    profile_cuda = cfg[PROFILER_KEY].get("CUDA", False)
    if profile_cpu:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    if profile_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    if len(activities) == 0:
        _warn("No activities specified, defaulting to CPU + CUDA")
        activities = _DEFAULT_PROFILER_ACTIVITIES

    # Set up profiler schedule
    schedule_cfg = cfg[PROFILER_KEY].get("schedule", None)

    # Use default schedule if None, else validate that schedule is valid and can be passed to `instantiate`
    if schedule_cfg is None:
        world_size, _ = get_world_size_and_rank()
        if world_size > 1:
            default_schedule_cfg = OmegaConf.create(_DEFAULT_SCHEDULE_DISTRIBUTED)
        else:
            default_schedule_cfg = OmegaConf.create(_DEFAULT_SCHEDULE_SINGLE)
        _warn(
            f" No schedule found in profiler config, loading default schedule {default_schedule_cfg}"
        )
        schedule_cfg = default_schedule_cfg
    else:
        if not all(k in schedule_cfg for k in ["wait", "warmup", "active"]):
            raise ValueError(
                "Invalid schedule config: must specify wait, warmup, active"
            )
        if "repeat" not in schedule_cfg:
            _warn(
                """ No repeat found in schedule config, setting to 1 (one cycle).
                If you want to cycle continuously, specify repeat = 0"""
            )
            schedule_cfg["repeat"] = 1
    if "_component_" not in schedule_cfg:
        schedule_cfg["_component_"] = "torch.profiler.schedule"
    schedule = config.instantiate(schedule_cfg)

    profile_memory = torch_profiler_cfg.get(
        "profile_memory", _DEFAULT_PROFILER_OPTS["profile_memory"]
    )

    # profile_memory requires with_stack and record_shapes, hence we override these if profile_memory is True
    # See torch.profiler.profiler._memory_profile
    with_stack = (
        torch_profiler_cfg.get("with_stack", _DEFAULT_PROFILER_OPTS["with_stack"])
        or profile_memory
    )
    record_shapes = (
        torch_profiler_cfg.get("record_shapes", _DEFAULT_PROFILER_OPTS["record_shapes"])
        or profile_memory
    )
    with_flops = torch_profiler_cfg.get(
        "with_flops", _DEFAULT_PROFILER_OPTS["with_flops"]
    )

    # experimental config is needed to export stacks: see https://github.com/pytorch/pytorch/issues/100253
    experimental_config = _ExperimentalConfig(verbose=True) if with_stack else None

    # Handle exporting of trace, memory timeline and other profiler artifacts
    profiler_output_dir = cfg[PROFILER_KEY].get("output_dir", None)

    if profiler_output_dir is None:
        _warn(
            f" No output directory found in profiler config, defaulting to {_DEFAULT_PROFILE_DIR}"
        )
        profiler_output_dir = _DEFAULT_PROFILE_DIR

    output_dir = Path(profiler_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    callback = partial(trace_handler, output_dir=output_dir)

    # Update profiler cfg in-place
    cfg[PROFILER_KEY].output_dir = profiler_output_dir
    cfg[PROFILER_KEY].schedule = schedule_cfg
    cfg[PROFILER_KEY].profile.profile_memory = profile_memory
    cfg[PROFILER_KEY].profile.with_stack = with_stack
    cfg[PROFILER_KEY].profile.record_shapes = record_shapes
    cfg[PROFILER_KEY].profile.with_flops = with_flops

    if "_component_" not in torch_profiler_cfg:
        cfg[PROFILER_KEY].profile["_component_"] = "torch.profiler.profile"

    profiler = config.instantiate(
        cfg[PROFILER_KEY].profile,
        activities=activities,
        schedule=schedule,
        experimental_config=experimental_config,
        on_trace_ready=callback,
    )

    return profiler
