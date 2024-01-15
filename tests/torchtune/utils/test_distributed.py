# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
from torch.distributed import launcher
from torchtune.utils.device import get_device
from torchtune.utils.distributed import (
    _get_process_group_backend_from_device,
    init_distributed,
)

from tests.test_utils import get_pet_launch_config


class TestDistributed:
    @staticmethod
    def _test_worker_fn(init_pg_explicit: bool) -> torch.device:
        """
        Integration test to confirm distributed initialization and consistency with process group backend utilities.
        """
        if init_pg_explicit:
            torch.distributed.init_process_group(backend="gloo")
        device = get_device()
        init_distributed(device=device)
        if not torch.distributed.is_initialized():
            raise AssertionError("Expected torch.distributed to be initialized")
        pg_backend = torch.distributed.get_backend()
        expected_pg_backend = (
            _get_process_group_backend_from_device(device)
            if not init_pg_explicit
            else "gloo"
        )
        if pg_backend != expected_pg_backend:
            raise AssertionError(
                f"Expected different process group backend: received {pg_backend}, expected {expected_pg_backend}"
            )
        return device

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

    def test_get_process_group_backend_cpu(self) -> None:
        device = torch.device("cpu")
        pg_backend = _get_process_group_backend_from_device(device)
        assert pg_backend == "gloo"

    def test_get_process_group_backend_gpu(self) -> None:
        device = torch.device("cuda:0")
        pg_backend = _get_process_group_backend_from_device(device)
        assert pg_backend == "nccl"
