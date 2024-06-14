# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch._C._profiler import _ExperimentalConfig

from torchtune.utils import (
    DEFAULT_TRACE_OPTS,
    FakeProfiler,
    PROFILER_KEY,
    setup_torch_profiler,
)

PROFILER_ATTRS = [
    "activities",
    "profile_memory",
    "with_stack",
    "record_shapes",
    "with_flops",
]


@pytest.fixture
def profiler_cfg():
    return """
profiler:
 enabled: True
 CPU: True
 CUDA: True
 profile_memory: False
 with_stack: False
 record_shapes: True
 with_flops: True
 schedule:
   wait: 3
   warmup: 1
   active: 1
   repeat: 0
"""


# This is a reference implementation of a profiler setup method to be defined within a `recipe`.
# A version of this lives in `torch.utils._profiler` but is not exported as the public API.
# Rather, the user is expected to define their own high-level setup function that parses the `cfg`
# and call a user-facing profiler setup function (e.g. `setup_torch_profiler`).
def _setup_profiler(
    cfg_profiler: DictConfig, return_cfg: bool = False
) -> torch.profiler.profile:
    """
    Parses the `profiler` section of top-level `cfg` and sets up profiler

    Args:
        cfg: DictConfig - `profiler` section of the top-level `cfg` (the main config passed to `recipe.main`)

    Returns:
        profiler: torch.profiler.profile | FakeProfiler - FakeProfiler is a nullcontext with no-op methods
        for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
        that the instrumented training loop does not need to be changed profiling is disabled.
    """
    # Check whether `profiler` key is present in the config and that it is not empty;
    # if it is present check that `enabled = True`
    if (cfg_profiler is not None and len(cfg_profiler) > 0) and cfg_profiler.get(
        "enabled", True
    ):
        enabled = True
    else:
        return FakeProfiler(), DictConfig({})

    # Parse profiler cfg
    cpu = cfg_profiler.get("CPU", False)
    cuda = cfg_profiler.get("CUDA", False)
    profile_memory = cfg_profiler.get(
        "profile_memory", DEFAULT_TRACE_OPTS["profile_memory"]
    )
    with_stack = cfg_profiler.get("with_stack", DEFAULT_TRACE_OPTS["with_stack"])
    record_shapes = cfg_profiler.get(
        "record_shapes", DEFAULT_TRACE_OPTS["record_shapes"]
    )
    with_flops = cfg_profiler.get("with_flops", DEFAULT_TRACE_OPTS["with_flops"])
    output_dir = cfg_profiler.get("output_dir", None)

    # Parse schedule specific args
    schedule_cfg = cfg_profiler.get("schedule", None)

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
    profiler, profiler_cfg = setup_torch_profiler(
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
    )
    return profiler, profiler_cfg


@pytest.fixture
def reference_profiler_basic():
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=3, warmup=1, active=1, repeat=0),
        profile_memory=False,
        with_stack=False,
        record_shapes=True,
        with_flops=True,
    )


@pytest.fixture
def reference_profiler_full():
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=3, warmup=1, active=1, repeat=0),
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
        with_flops=True,
        experimental_config=_ExperimentalConfig(verbose=True),
    )


def check_profiler_attrs(profiler, ref_profiler):
    for attr in PROFILER_ATTRS:
        assert getattr(profiler, attr) == getattr(ref_profiler, attr)


def check_schedule(schedule, ref_schedule, num_steps=10):
    ref_steps = [ref_schedule(i) for i in range(num_steps)]
    test_steps = [schedule(i) for i in range(num_steps)]
    assert ref_steps == test_steps


def parse_scheduler_cfg(schedule_cfg: DictConfig):
    return {
        k: schedule_cfg.get(k, None) for k in ["wait", "warmup", "active", "repeat"]
    }


def test_instantiate_basic(profiler_cfg, reference_profiler_basic):
    cfg = OmegaConf.create(profiler_cfg)

    # Check that schedule can be instantiated correctly
    schedule_cfg = cfg[PROFILER_KEY].schedule
    test_schedule = torch.profiler.schedule(**parse_scheduler_cfg(schedule_cfg))
    ref_schedule = reference_profiler_basic.schedule
    check_schedule(ref_schedule, test_schedule)

    test_activities = []
    if cfg[PROFILER_KEY].CPU:
        test_activities.append(torch.profiler.ProfilerActivity.CPU)
    if cfg[PROFILER_KEY].CUDA:
        test_activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_cfg = {
        k: cfg[PROFILER_KEY].get(k, None) for k in PROFILER_ATTRS if k != "activities"
    }
    test_profiler = torch.profiler.profile(
        activities=test_activities, schedule=test_schedule, **trace_cfg
    )
    check_profiler_attrs(test_profiler, reference_profiler_basic)


def test_instantiate_full(profiler_cfg, reference_profiler_full):
    cfg = OmegaConf.create(profiler_cfg)[PROFILER_KEY]

    # Check `setup` automatically overrides `with_stack` and `record_shapes` when profile_memory is True and adds
    # experimental_config, which is needed for stack exporting (see comments in `setup_torch_profiler`)
    cfg.profile_memory = True
    cfg.with_stack = False
    cfg.record_shapes = False
    profiler, updated_cfg = _setup_profiler(cfg)

    check_profiler_attrs(profiler, reference_profiler_full)
    assert profiler.experimental_config is not None
    assert updated_cfg.with_stack is True
    assert updated_cfg.record_shapes is True


def test_schedule_setup(profiler_cfg, reference_profiler_basic):
    from torchtune.utils._profiler import _DEFAULT_SCHEDULE

    cfg = OmegaConf.create(profiler_cfg)[PROFILER_KEY]

    # Test that after removing schedule, setup method will implement default schedule
    cfg.pop("schedule")
    profiler, updated_cfg = _setup_profiler(cfg)
    test_schedule = profiler.schedule
    ref_schedule = torch.profiler.schedule(**_DEFAULT_SCHEDULE)
    check_schedule(ref_schedule, test_schedule)
    for k in ["wait", "warmup", "active", "repeat"]:
        assert updated_cfg.schedule[k] == _DEFAULT_SCHEDULE[k]

    # Test invalid schedule (invalid defined as any of wait, warmup, active missing)
    for k in ["wait", "warmup", "active"]:
        cfg = OmegaConf.create(profiler_cfg)[PROFILER_KEY]
        cfg.schedule.pop(k)
        with pytest.raises(ValueError):
            profiler, _ = _setup_profiler(cfg)

    # Test repeat is set to 1 if missing but all other schedule keys are present
    cfg = OmegaConf.create(profiler_cfg)[PROFILER_KEY]
    cfg.schedule.pop("repeat")
    profiler, updated_cfg = _setup_profiler(cfg)
    test_schedule = profiler.schedule
    ref_schedule = torch.profiler.schedule(
        wait=cfg.schedule.wait,
        warmup=cfg.schedule.warmup,
        active=cfg.schedule.active,
        repeat=1,
    )
    num_steps_per_cycle = cfg.schedule.wait + cfg.schedule.warmup + cfg.schedule.active
    # Repeat means only 1 cycle, hence we check 2 cycles
    check_schedule(ref_schedule, test_schedule, num_steps=2 * num_steps_per_cycle)
    assert updated_cfg.schedule.repeat == 1


def test_default_activities(profiler_cfg):
    cfg = OmegaConf.create(profiler_cfg)[PROFILER_KEY]

    from torchtune.utils._profiler import _DEFAULT_PROFILER_ACTIVITIES

    # Test setup automatically adds CPU + CUDA tracing if neither CPU nor CUDA is specified
    cfg.pop("CPU")
    cfg.pop("CUDA")
    profiler, updated_cfg = _setup_profiler(cfg)
    assert profiler.activities == _DEFAULT_PROFILER_ACTIVITIES
    assert updated_cfg.CPU is True
    assert updated_cfg.CUDA is True


def test_default_output_dir(profiler_cfg):
    cfg = OmegaConf.create(profiler_cfg)[PROFILER_KEY]
    from torchtune.utils._profiler import _DEFAULT_PROFILE_DIR

    # Test cfg output_dir is set correctly
    if cfg.get("output_dir", None) is not None:
        cfg.pop("output_dir")
    _, updated_cfg = _setup_profiler(cfg, return_cfg=True)
    assert updated_cfg.output_dir == _DEFAULT_PROFILE_DIR


def test_default_trace_opts(profiler_cfg):
    cfg = OmegaConf.create(profiler_cfg)[PROFILER_KEY]
    from torchtune.utils._profiler import _DEFAULT_PROFILER_ACTIVITIES

    # Test missing profiler options are set to defaults
    cfg.pop("profile_memory")
    cfg.pop("with_stack")
    cfg.pop("record_shapes")
    cfg.pop("with_flops")
    profiler, updated_cfg = _setup_profiler(cfg)
    check_profiler_attrs(
        profiler,
        torch.profiler.profile(
            activities=_DEFAULT_PROFILER_ACTIVITIES, **DEFAULT_TRACE_OPTS
        ),
    )
    for k in ["profile_memory", "with_stack", "record_shapes", "with_flops"]:
        assert updated_cfg[k] == DEFAULT_TRACE_OPTS[k]


def test_fake_profiler(profiler_cfg):

    # Test missing `profile` key returns fake profiler
    cfg = OmegaConf.create(profiler_cfg)
    cfg.pop(PROFILER_KEY)
    profiler, _ = _setup_profiler(cfg)
    assert isinstance(profiler, FakeProfiler)

    # Test that disabled profiler creates fake profiler
    cfg = OmegaConf.create(profiler_cfg)[PROFILER_KEY]
    cfg.enabled = False
    profiler, _ = _setup_profiler(cfg)
    assert isinstance(profiler, FakeProfiler)

    # Test that fake_profiler.step() does nothing both when used as context manager and as standalone object
    with profiler as prof:
        prof.step()

    # Additional FakeProfiler no-ops when used as object and not context
    assert profiler.step() is None
    assert profiler.start() is None
    assert profiler.stop() is None
