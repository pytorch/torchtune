# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from torch.distributed import launcher

# from torchtune.utils.device import get_device
from torchtune.utils.distributed import get_world_size_and_rank, init_distributed

from tests.test_utils import get_pet_launch_config


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
            init_distributed()
        if not torch.distributed.is_initialized():
            raise AssertionError("Expected torch.distributed to be initialized")
        pg_backend = torch.distributed.get_backend()
        expected_pg_backend = "undefined" if not init_pg_explicit else "gloo"
        if pg_backend != expected_pg_backend:
            raise AssertionError(
                f"Expected different process group backend: received {pg_backend}, expected {expected_pg_backend}"
            )

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


# TODO: Add FSDP specific tests and _broadcast and _is_distributed
