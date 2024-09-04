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
from torchtune.training.seed import set_seed


class TestSeed:
    def test_seed_range(self) -> None:
        """
        Verify that exceptions are raised on input values
        """
        with pytest.raises(ValueError, match="Invalid seed value provided"):
            set_seed(-1)

        invalid_max = np.iinfo(np.uint64).max
        with pytest.raises(ValueError, match="Invalid seed value provided"):
            set_seed(invalid_max)

        # should not raise any exceptions
        set_seed(42)

    def test_deterministic_true(self) -> None:
        for det_debug_mode, det_debug_mode_str in [(1, "warn"), (2, "error")]:
            warn_only = det_debug_mode == 1
            for debug_mode in (det_debug_mode, det_debug_mode_str):
                # torch/testing/_internal/common_utils.py calls `disable_global_flags()`
                # workaround RuntimeError: not allowed to set ... after disable_global_flags
                setattr(  # noqa: B010
                    torch.backends, "__allow_nonbracketed_mutation_flag", True
                )
                set_seed(42, debug_mode=debug_mode)
                setattr(  # noqa: B010
                    torch.backends, "__allow_nonbracketed_mutation_flag", False
                )
                assert torch.backends.cudnn.deterministic
                assert not torch.backends.cudnn.benchmark
                assert det_debug_mode == torch.get_deterministic_debug_mode()
                assert torch.are_deterministic_algorithms_enabled()
                assert (
                    warn_only == torch.is_deterministic_algorithms_warn_only_enabled()
                )
                assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    def test_deterministic_false(self) -> None:
        for debug_mode in ("default", 0):
            setattr(  # noqa: B010
                torch.backends, "__allow_nonbracketed_mutation_flag", True
            )
            set_seed(42, debug_mode=debug_mode)
            setattr(  # noqa: B010
                torch.backends, "__allow_nonbracketed_mutation_flag", False
            )
            assert not torch.backends.cudnn.deterministic
            assert torch.backends.cudnn.benchmark
            assert 0 == torch.get_deterministic_debug_mode()
            assert not torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()

    def test_deterministic_unset(self) -> None:
        det = torch.backends.cudnn.deterministic
        benchmark = torch.backends.cudnn.benchmark
        det_debug_mode = torch.get_deterministic_debug_mode()
        det_algo_enabled = torch.are_deterministic_algorithms_enabled()
        det_algo_warn_only_enabled = (
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
        set_seed(42, debug_mode=None)
        assert det == torch.backends.cudnn.deterministic
        assert benchmark == torch.backends.cudnn.benchmark
        assert det_debug_mode == torch.get_deterministic_debug_mode()
        assert det_algo_enabled == torch.are_deterministic_algorithms_enabled()
        assert (
            det_algo_warn_only_enabled
            == torch.is_deterministic_algorithms_warn_only_enabled()
        )
