# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
import torch.nn as nn
from torch.distributed import launcher

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torchtune.utils.distributed import (
    get_world_size_and_rank,
    init_distributed,
    wrap_fsdp,
)

from tests.test_utils import get_pet_launch_config, single_box_init


class TestDistributed:
    def test_init_distributed(self) -> None:
        """Integration test to confirm consistency across device initialization utilities."""
        distributed = init_distributed()
        assert (
            not distributed
        ), "Should return False as there are no distributed environment variables"

    @staticmethod
    def _test_worker_fn(init_pg_explicit: bool) -> None:
        """
        Integration test to confirm distributed initialization and consistency with process group backend utilities.
        """
        if init_pg_explicit:
            torch.distributed.init_process_group(backend="gloo")
        if not torch.distributed.is_initialized():
            init_distributed(backend="gloo")
        if not torch.distributed.is_initialized():
            raise AssertionError("Expected torch.distributed to be initialized")
        pg_backend = torch.distributed.get_backend()
        assert (
            pg_backend == "gloo"
        ), f"Expected 'gloo' backend, but received {pg_backend}"

    @staticmethod
    def _test_world_size_with_cpu_device(expected_world_size: int) -> None:
        init_distributed(backend="gloo")
        world_size, _ = get_world_size_and_rank()
        if world_size != expected_world_size:
            raise AssertionError(
                f"Expected different world size: received {world_size}, expected {expected_world_size}"
            )

    def _test_launch_worker(
        self,
        num_processes: int,
        init_pg_explicit: bool,
    ) -> None:
        lc = get_pet_launch_config(num_processes)
        launcher.elastic_launch(lc, entrypoint=self._test_worker_fn)(init_pg_explicit)

    def test_init_from_env_no_dup(self) -> None:
        self._test_launch_worker(2, init_pg_explicit=False)
        # trivial test case to ensure test passes with no exceptions
        assert True

    def test_init_from_env_dup(self) -> None:
        self._test_launch_worker(2, init_pg_explicit=True)
        # trivial test case to ensure test passes with no exceptions
        assert True

    def test_world_size_with_cpu(self) -> None:
        desired_world_size = 4
        lc = get_pet_launch_config(desired_world_size)
        launcher.elastic_launch(lc, entrypoint=self._test_world_size_with_cpu_device)(
            desired_world_size
        )

    def test_default_wrap_fsdp(self) -> None:
        with single_box_init():
            model = nn.Linear(5, 5)
            fsdp_model = wrap_fsdp(
                model, device=torch.device("cpu"), dtype=torch.float32
            )
            # Should create a single FSDP unit with FULL_SHARD
            fsdp_units = [m for m in fsdp_model.modules() if isinstance(m, FSDP)]
            assert len(fsdp_units) == 1

    def test_wrap_fsdp_wrapping(self) -> None:
        with single_box_init():
            model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
            orig_num_modules = len([m for m in model.modules()])
            fsdp_model = wrap_fsdp(
                model,
                device=torch.device("cpu"),
                dtype=torch.float32,
                auto_wrap_policy={nn.Linear},
            )
            # Should create orig_num_modules FSDP units.
            fsdp_units = [m for m in fsdp_model.modules() if isinstance(m, FSDP)]
            assert len(fsdp_units) == orig_num_modules

    def test_wrap_fsdp_custom_policy(self) -> None:
        def always_wrap(*args, **kwargs):
            return True

        model = nn.Sequential(
            nn.Linear(3, 3), nn.BatchNorm1d(10), nn.Dropout(0.25), nn.Softmax(dim=1)
        )
        num_modules = len([m for m in model.modules()])
        with single_box_init():
            fsdp_model = wrap_fsdp(
                model,
                device=torch.device("cpu"),
                dtype=torch.float32,
                auto_wrap_policy=always_wrap,
            )
            fsdp_units = [m for m in fsdp_model.modules() if isinstance(m, FSDP)]
            assert len(fsdp_units) == num_modules
