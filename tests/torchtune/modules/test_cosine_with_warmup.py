# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest

import torch
import torch.optim as optim

from torchtune.lr_schedulers.cosine_with_warmup import cosine_schedule_with_warmup

from tests.test_utils import assert_expected


class TestCosineLR:
    @pytest.fixture
    def scheduler(self):
        optimizer = optim.SGD([torch.ones(1)], lr=0.2)
        scheduler = cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
            num_cycles=1.0,
        )
        return scheduler

    def test_cosine_schedule_init(self, scheduler):
        optimizer = scheduler.optimizer
        assert_expected(optimizer.param_groups[0]["lr"], 0.0)

    def test_cosine_schedule_mid_warmup(self, scheduler):
        optimizer = scheduler.optimizer
        scheduler.last_epoch = 5 - 1
        optimizer.step()
        scheduler.step()
        assert_expected(optimizer.param_groups[0]["lr"], 0.1)

    def test_cosine_schedule_warmup(self, scheduler):
        optimizer = scheduler.optimizer
        scheduler.last_epoch = 10 - 1
        optimizer.step()
        scheduler.step()
        assert_expected(optimizer.param_groups[0]["lr"], 0.2)

    def test_cosine_schedule_minimum_value(self, scheduler):
        optimizer = scheduler.optimizer
        scheduler.last_epoch = 55 - 1
        optimizer.step()
        scheduler.step()
        assert_expected(optimizer.param_groups[0]["lr"], 0.0)

    def test_cosine_schedule_complete_cycle(self, scheduler):
        optimizer = scheduler.optimizer
        scheduler.last_epoch = 100 - 1
        optimizer.step()
        scheduler.step()
        assert_expected(optimizer.param_groups[0]["lr"], 0.2)
