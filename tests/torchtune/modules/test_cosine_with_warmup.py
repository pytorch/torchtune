# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest

import torch
import torch.optim as optim

from tests.test_utils import assert_expected

from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup


class TestCosineLR:
    @pytest.fixture
    def scheduler(self):
        optimizer = optim.SGD([torch.ones(1)], lr=0.2)
        scheduler = get_cosine_schedule_with_warmup(
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


class TestCosineLRWithMinLr:
    @pytest.fixture
    def scheduler(self):
        optimizer = optim.SGD([torch.ones(1)], lr=0.2)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
            min_lr_warmup=0.02,
            min_lr_decay=0.01,
            num_cycles=1.0,
        )
        return scheduler

    def test_cosine_schedule_init(self, scheduler):
        optimizer = scheduler.optimizer
        assert_expected(optimizer.param_groups[0]["lr"], 0.02)

    def test_cosine_schedule_mid_warmup(self, scheduler):
        optimizer = scheduler.optimizer
        scheduler.last_epoch = 5 - 1
        optimizer.step()
        scheduler.step()
        assert_expected(optimizer.param_groups[0]["lr"], 0.11)

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
        assert_expected(optimizer.param_groups[0]["lr"], 0.01)

    def test_cosine_schedule_complete_cycle(self, scheduler):
        optimizer = scheduler.optimizer
        scheduler.last_epoch = 100 - 1
        optimizer.step()
        scheduler.step()
        assert_expected(optimizer.param_groups[0]["lr"], 0.2)
