# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from torch._C._profiler import _ExperimentalConfig

from torchtune import config
from torchtune.utils._profiler import PROFILER_KEY, setup_torch_profiler

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
 profile:
   # _component_: torch.profiler.profile
   profile_memory: False
   with_stack: False
   record_shapes: True
   with_flops: True
 schedule:
   # _component_: torch.profiler.schedule
   wait: 3
   warmup: 1
   active: 1
   repeat: 0
"""


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
    cfg = OmegaConf.create(profiler_cfg)

    torch_profiler_cfg = cfg[PROFILER_KEY].profile
    if "_component_" not in torch_profiler_cfg:
        torch_profiler_cfg["_component_"] = "torch.profiler.profile"
    schedule_cfg = cfg[PROFILER_KEY].schedule
    if "_component_" not in schedule_cfg:
        schedule_cfg["_component_"] = "torch.profiler.schedule"

    ref_schedule = torch.profiler.schedule(wait=3, warmup=1, active=1, repeat=0)
    test_schedule = config.instantiate(schedule_cfg)
    check_schedule(ref_schedule, test_schedule)

    test_activities = []
    if cfg[PROFILER_KEY].CPU:
        test_activities.append(torch.profiler.ProfilerActivity.CPU)
    if cfg[PROFILER_KEY].CUDA:
        test_activities.append(torch.profiler.ProfilerActivity.CUDA)
    test_profiler = config.instantiate(
        torch_profiler_cfg, activities=test_activities, schedule=test_schedule
    )
    check_profiler_attrs(test_profiler, reference_profiler_basic)


def test_instantiate_full(profiler_cfg, reference_profiler_full):
    cfg = OmegaConf.create(profiler_cfg)

    # Check `setup` automatically overrides `with_stack` and `record_shapes` when profile_memory is True and adds
    # experimental_config, which is needed for stack exporting (see comments in `setup_torch_profiler`)
    cfg[PROFILER_KEY].profile.profile_memory = True
    cfg[PROFILER_KEY].profile.with_stack = False
    cfg[PROFILER_KEY].profile.record_shapes = False
    profiler = setup_torch_profiler(cfg)

    check_profiler_attrs(profiler, reference_profiler_full)
    assert profiler.experimental_config is not None


def test_schedule_setup(profiler_cfg, reference_profiler_basic):
    from torchtune.utils._profiler import (
        _DEFAULT_SCHEDULE_DISTRIBUTED,
        _DEFAULT_SCHEDULE_SINGLE,
    )

    cfg = OmegaConf.create(profiler_cfg)
    profiler = setup_torch_profiler(cfg)
    check_profiler_attrs(profiler, reference_profiler_basic)

    # Test that after removing schedule, setup method will implement default schedule
    with patch(
        "torchtune.utils._profiler.get_world_size_and_rank", return_value=(1, 0)
    ):
        cfg[PROFILER_KEY].pop("schedule")
        profiler = setup_torch_profiler(cfg)
        assert cfg[PROFILER_KEY].schedule == _DEFAULT_SCHEDULE_SINGLE

    with patch(
        "torchtune.utils._profiler.get_world_size_and_rank", return_value=(2, 0)
    ):
        cfg[PROFILER_KEY].pop("schedule")
        profiler = setup_torch_profiler(cfg)
        assert cfg[PROFILER_KEY].schedule == _DEFAULT_SCHEDULE_DISTRIBUTED

    # Test invalid schedule
    cfg[PROFILER_KEY].schedule.pop("wait")
    with pytest.raises(ValueError):
        profiler = setup_torch_profiler(cfg)


def test_default_activities(profiler_cfg):
    cfg = OmegaConf.create(profiler_cfg)

    from torchtune.utils._profiler import _DEFAULT_PROFILER_ACTIVITIES

    # Test setup automatically adds CPU + CUDA tracing if neither CPU nor CUDA is specified
    cfg[PROFILER_KEY].pop("CPU")
    cfg[PROFILER_KEY].pop("CUDA")
    profiler = setup_torch_profiler(cfg)
    assert profiler.activities == _DEFAULT_PROFILER_ACTIVITIES


def test_default_output_dir(profiler_cfg):
    cfg = OmegaConf.create(profiler_cfg)
    from torchtune.utils._profiler import _DEFAULT_PROFILE_DIR

    # Test cfg output_dir is set correctly
    if cfg[PROFILER_KEY].get("output_dir", None) is not None:
        cfg[PROFILER_KEY].pop("output_dir")
    _profiler = setup_torch_profiler(cfg)
    assert cfg[PROFILER_KEY].output_dir == _DEFAULT_PROFILE_DIR


def test_default_profiler(profiler_cfg):
    cfg = OmegaConf.create(profiler_cfg)

    from torchtune.utils._profiler import _DEFAULT_PROFILER_OPTS

    # Test missing profiler options are set to defaults
    torch_profiler_cfg = cfg[PROFILER_KEY].profile
    torch_profiler_cfg.pop("profile_memory")
    torch_profiler_cfg.pop("with_stack")
    torch_profiler_cfg.pop("record_shapes")
    torch_profiler_cfg.pop("with_flops")
    profiler = setup_torch_profiler(cfg)
    assert torch_profiler_cfg.profile_memory == _DEFAULT_PROFILER_OPTS["profile_memory"]
    assert torch_profiler_cfg.with_stack == _DEFAULT_PROFILER_OPTS["with_stack"]
    assert torch_profiler_cfg.record_shapes == _DEFAULT_PROFILER_OPTS["record_shapes"]
    assert torch_profiler_cfg.with_flops == _DEFAULT_PROFILER_OPTS["with_flops"]

    # Test missing torch profiler entirely
    cfg[PROFILER_KEY].pop("profile")
    profiler = setup_torch_profiler(cfg)
    assert cfg[PROFILER_KEY].profile is not None
    assert cfg[PROFILER_KEY].profile == OmegaConf.create(
        {"_component_": "torch.profiler.profile", **_DEFAULT_PROFILER_OPTS}
    )


def test_fake_profiler(profiler_cfg):
    from torchtune.utils._profiler import FakeProfiler, PROFILER_KEY

    # Test that disabled profiler creates fake profiler
    cfg = OmegaConf.create(profiler_cfg)
    cfg[PROFILER_KEY].enabled = False

    profiler = setup_torch_profiler(cfg)
    assert isinstance(profiler, FakeProfiler)

    # Test that fake_profiler.step() does nothing both when used as context manager and as standalone object
    with profiler as prof:
        prof.step()
    assert profiler.step() is None
    
    # Additional FakeProfiler no-ops
    assert profiler.start() is None
    assert profiler.stop() is None
    
    # Test missing `profile` key returns fake profiler
    cfg.pop(PROFILER_KEY)
    profiler = setup_torch_profiler(cfg)
    assert isinstance(profiler, FakeProfiler)
