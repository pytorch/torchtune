# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os

import numpy as np
import pytest
import torch
from torch.distributed import launcher
from torchtune.utils.device import _get_device_from_env
from torchtune.utils.env import (
    _get_process_group_backend_from_device,
    init_from_env,
    seed,
)

from tests.test_utils import get_pet_launch_config, skip_if_cuda_not_available


class TestEnv:
    @skip_if_cuda_not_available
    def test_init_from_env_non_zero_device(self) -> None:
        device = init_from_env(device_type="cuda:1")
        assert device == torch.device("cuda:1")
        device_idx = torch.cuda.current_device()
        set_device = torch.device(device_idx)
        assert set_device == device

    def test_init_from_env(self) -> None:
        """Integration test to confirm consistency across device initialization utilities."""
        device = init_from_env()
        assert device == _get_device_from_env()
        assert not torch.distributed.is_initialized()

    @staticmethod
    def _test_worker_fn(init_pg_explicit: bool) -> torch.device:
        """
        Integration test to confirm distributed initialization and consistency with process group backend utilities.
        """
        if init_pg_explicit:
            torch.distributed.init_process_group(backend="gloo")
        device = init_from_env()
        if not torch.distributed.is_initialized():
            raise AssertionError("Expected torch.distributed to be initialized")
        device_from_env = _get_device_from_env()
        if device != device_from_env:
            raise AssertionError(
                f"Expected different device: received {device}, expected {device_from_env}"
            )
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

    def test_seed_range(self) -> None:
        """
        Verify that exceptions are raised on input values
        """
        with pytest.raises(ValueError, match="Invalid seed value provided"):
            seed(-1)

        invalid_max = np.iinfo(np.uint64).max
        with pytest.raises(ValueError, match="Invalid seed value provided"):
            seed(invalid_max)

        # should not raise any exceptions
        seed(42)

    def test_debug_mode_true(self) -> None:
        for det_debug_mode, det_debug_mode_str in [(1, "warn"), (2, "error")]:
            warn_only = det_debug_mode == 1
            for debug_mode in (det_debug_mode, det_debug_mode_str):
                seed(42, debug_mode=debug_mode)
                assert torch.backends.cudnn.deterministic
                assert not torch.backends.cudnn.benchmark
                assert det_debug_mode == torch.get_deterministic_debug_mode()
                assert torch.are_deterministic_algorithms_enabled()
                assert (
                    warn_only == torch.is_deterministic_algorithms_warn_only_enabled()
                )
                assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    def test_debug_mode_false(self) -> None:
        for debug_mode in ("default", 0):
            seed(42, debug_mode=debug_mode)
            assert not torch.backends.cudnn.deterministic
            assert torch.backends.cudnn.benchmark
            assert 0 == torch.get_deterministic_debug_mode()
            assert not torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()

    def test_debug_mode_unset(self) -> None:
        det = torch.backends.cudnn.deterministic
        benchmark = torch.backends.cudnn.benchmark
        det_debug_mode = torch.get_deterministic_debug_mode()
        det_algo_enabled = torch.are_deterministic_algorithms_enabled()
        det_algo_warn_only_enabled = (
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
        seed(42, debug_mode=None)
        assert det == torch.backends.cudnn.deterministic
        assert benchmark == torch.backends.cudnn.benchmark
        assert det_debug_mode == torch.get_deterministic_debug_mode()
        assert det_algo_enabled == torch.are_deterministic_algorithms_enabled()
        assert (
            det_algo_warn_only_enabled
            == torch.is_deterministic_algorithms_warn_only_enabled()
        )
