# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch._C._profiler import _ExperimentalConfig
from torchtune import config
from torchtune.training import (
    DEFAULT_PROFILE_DIR,
    DEFAULT_PROFILER_ACTIVITIES,
    DEFAULT_SCHEDULE,
    DEFAULT_TRACE_OPTS,
    DummyProfiler,
    PROFILER_KEY,
)

# Disable logging otherwise output will be very verbose
logging.basicConfig(level=logging.ERROR)

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
 cpu: True
 cuda: True
 profile_memory: False
 with_stack: False
 record_shapes: True
 with_flops: True
 wait_steps: 3
 warmup_steps: 1
 active_steps: 1
 num_cycles: 0
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
        cfg_profiler (DictConfig): `profiler` section of the top-level `cfg` (the main config passed to `recipe.main`)
        return_cfg (bool): Doesn't seem to be used. Default False.

    Returns:
        profiler: torch.profiler.profile | DummyProfiler - DummyProfiler is a nullcontext with no-op methods
        for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
        that the instrumented training loop does not need to be changed profiling is disabled.
    """
    # Missing profiler section in config, assume disabled
    if cfg_profiler is None:
        cfg_profiler = DictConfig({"enabled": False})

    # Check that component is included and set correctly
    if cfg_profiler.get("_component_", None) is None:
        cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
    else:
        assert (
            cfg_profiler.get("_component_") == "torchtune.training.setup_torch_profiler"
        ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

    profiler, profiler_cfg = config.instantiate(cfg_profiler)

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


def test_instantiate_basic(profiler_cfg, reference_profiler_basic):
    cfg = OmegaConf.create(profiler_cfg)[PROFILER_KEY]

    profiler, updated_cfg = _setup_profiler(cfg)

    check_profiler_attrs(profiler, reference_profiler_basic)

    ref_schedule = torch.profiler.schedule(
        wait=updated_cfg["wait_steps"],
        warmup=updated_cfg["warmup_steps"],
        active=updated_cfg["active_steps"],
        repeat=updated_cfg["num_cycles"],
    )
    check_schedule(profiler.schedule, ref_schedule)


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

    cfg = OmegaConf.create(profiler_cfg)[PROFILER_KEY]

    # Test that after removing schedule, setup method will implement default schedule
    _ = [cfg.pop(k) for k in DEFAULT_SCHEDULE.keys()]
    profiler, updated_cfg = _setup_profiler(cfg)
    test_schedule = profiler.schedule
    ref_schedule = torch.profiler.schedule(
        wait=DEFAULT_SCHEDULE["wait_steps"],
        warmup=DEFAULT_SCHEDULE["warmup_steps"],
        active=DEFAULT_SCHEDULE["active_steps"],
        repeat=DEFAULT_SCHEDULE["num_cycles"],
    )
    check_schedule(ref_schedule, test_schedule)

    # Check cfg is updated correctly
    for k in DEFAULT_SCHEDULE.keys():
        assert updated_cfg[k] == DEFAULT_SCHEDULE[k]

    # Test missing key is automatically set to default
    for k in DEFAULT_SCHEDULE.keys():
        cfg = OmegaConf.create(profiler_cfg)[PROFILER_KEY]
        cfg.pop(k)
        profiler, updated_cfg = _setup_profiler(cfg)
        assert updated_cfg[k] == DEFAULT_SCHEDULE[k]


def test_default_activities(profiler_cfg):
    cfg = OmegaConf.create(profiler_cfg)[PROFILER_KEY]

    # Test setup automatically adds CPU + CUDA tracing if neither CPU nor CUDA is specified
    cfg.pop("cpu")
    cfg.pop("cuda")
    profiler, updated_cfg = _setup_profiler(cfg)
    assert profiler.activities == DEFAULT_PROFILER_ACTIVITIES
    assert updated_cfg.cpu is True
    assert updated_cfg.cuda is True


def test_default_output_dir(profiler_cfg):
    cfg = OmegaConf.create(profiler_cfg)[PROFILER_KEY]

    # Test cfg output_dir is set correctly
    if cfg.get("output_dir", None) is not None:
        cfg.pop("output_dir")
    _, updated_cfg = _setup_profiler(cfg, return_cfg=True)
    assert updated_cfg.output_dir == DEFAULT_PROFILE_DIR


def test_default_trace_opts(profiler_cfg):
    cfg = OmegaConf.create(profiler_cfg)[PROFILER_KEY]

    # Test missing profiler options are set to defaults
    cfg.pop("profile_memory")
    cfg.pop("with_stack")
    cfg.pop("record_shapes")
    cfg.pop("with_flops")
    profiler, updated_cfg = _setup_profiler(cfg)
    check_profiler_attrs(
        profiler,
        torch.profiler.profile(
            activities=DEFAULT_PROFILER_ACTIVITIES, **DEFAULT_TRACE_OPTS
        ),
    )
    for k in ["profile_memory", "with_stack", "record_shapes", "with_flops"]:
        assert updated_cfg[k] == DEFAULT_TRACE_OPTS[k]


def test_dummy_profiler(profiler_cfg):

    # Test missing `profile` key returns fake profiler
    cfg = OmegaConf.create(profiler_cfg)
    cfg.pop(PROFILER_KEY)
    profiler, _ = _setup_profiler(cfg)
    assert isinstance(profiler, DummyProfiler)

    # Test that disabled profiler creates fake profiler
    cfg = OmegaConf.create(profiler_cfg)[PROFILER_KEY]
    cfg.enabled = False
    profiler, _ = _setup_profiler(cfg)
    assert isinstance(profiler, DummyProfiler)

    # Test that fake_profiler.step() does nothing both when used as context manager and as standalone object
    with profiler as prof:
        prof.step()

    # Additional DummyProfiler no-ops when used as object and not context
    assert profiler.step() is None
    assert profiler.start() is None
    assert profiler.stop() is None
