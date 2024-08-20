# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import gc

import pytest
import torch
from torch import nn
from torchtune.modules.optim import CPUOffloadOptimizer
from torchtune.modules.optim.cpu_offload import _CPU_OFFLOAD_OPTIM_AVAILABLE

if not _CPU_OFFLOAD_OPTIM_AVAILABLE:
    pytest.skip("CPU offload optimizer is not available", allow_module_level=True)


class TestCPUOffloadOptimizer:
    def _reset_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def _get_model(self, dim: int):
        return nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CPU offload optimizer requires CUDA"
    )
    @pytest.mark.parametrize("offload_gradients", [False, True])
    def test_cpu_offload_optimizer_correctness(self, offload_gradients):
        dim = 16
        model_ref = self._get_model(dim).cuda()
        optim_ref = torch.optim.AdamW(model_ref.parameters())

        model_test = copy.deepcopy(model_ref)
        optim_test = CPUOffloadOptimizer(
            model_test.parameters(),
            optimizer_class="torch.optim.AdamW",
            offload_gradients=offload_gradients,
        )

        for _ in range(4):
            x = torch.randn(4, dim, device="cuda")
            loss_ref = model_ref(x).mean()
            loss_test = model_test(x).mean()

            torch.testing.assert_close(loss_ref, loss_test)

            optim_ref.zero_grad()
            loss_ref.backward()
            optim_ref.step()

            optim_test.zero_grad()
            loss_test.backward()
            optim_test.step()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CPU offload optimizer requires CUDA"
    )
    @pytest.mark.parametrize("offload_gradients", [False, True])
    def test_cpu_offload_optimizer_memory(self, offload_gradients):
        dim = 16
        self._reset_memory()
        model_ref = self._get_model(dim).cuda()
        optim_ref = torch.optim.AdamW(model_ref.parameters())

        for _ in range(4):
            x = torch.randn(4, dim, device="cuda")
            optim_ref.zero_grad()
            model_ref(x).mean().backward()
            optim_ref.step()

        memory_ref = torch.cuda.max_memory_allocated()
        del model_ref
        del optim_ref
        self._reset_memory()

        model_test = self._get_model(dim).cuda()
        optim_test = CPUOffloadOptimizer(
            model_test.parameters(),
            optimizer_class="torch.optim.AdamW",
            offload_gradients=offload_gradients,
        )

        for _ in range(4):
            x = torch.randn(4, dim, device="cuda")
            optim_test.zero_grad()
            model_test(x).mean().backward()
            optim_test.step()

        memory_test = torch.cuda.max_memory_allocated()
        assert memory_test < memory_ref
