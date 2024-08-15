# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
from torch import nn
from torchtune.modules.optim import CPUOffloadOptimizer
from torchtune.modules.optim._cpu_offload import _CPU_OFFLOAD_OPTIM_AVAILABLE

if not _CPU_OFFLOAD_OPTIM_AVAILABLE:
    pytest.skip("CPU offload optimizer is not available", allow_module_level=True)


class TestCPUOffloadOptimizer:
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CPU offload optimizer requires CUDA"
    )
    @pytest.mark.parametrize("offload_gradients", [False, True])
    def test_cpu_offload_optimizer(self, offload_gradients):
        model_ref = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128)
        ).cuda()
        optim_ref = torch.optim.AdamW(model_ref.parameters())

        model_test = copy.deepcopy(model_ref)
        optim_test = CPUOffloadOptimizer(
            model_test.parameters(),
            optimizer_class="torch.optim.AdamW",
            offload_gradients=offload_gradients,
        )

        for _ in range(4):
            x = torch.randn(4, 128, device="cuda")
            loss_ref = model_ref(x).mean()
            loss_test = model_test(x).mean()

            torch.testing.assert_close(loss_ref, loss_test)

            optim_ref.zero_grad()
            loss_ref.backward()
            optim_ref.step()

            optim_test.zero_grad()
            loss_test.backward()
            optim_test.step()
