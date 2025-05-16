# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# test_optim.py
import pytest
import torch

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchtune.modules.optim import OptimizerInBackward


class TestOptimizerInBackward:
    @pytest.fixture
    def dummy_model(self):
        return nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
        )

    @pytest.fixture
    def dummy_input_and_target(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.randn(10, 10), torch.randn(10, 10)

    def test_adamw(self, dummy_model, dummy_input_and_target):
        """Basic test with AdamW optimizer just to confirm it runs"""
        optimizer = OptimizerInBackward(
            dummy_model.parameters(),
            AdamW,
            lr=0.1,
        )
        initial_optimizer_sd = optimizer.state_dict()
        optimizer.zero_grad()
        output = dummy_model(dummy_input_and_target[0])
        loss = nn.functional.mse_loss(output, dummy_input_and_target[1])
        loss.backward()
        post_loss_optimizer_sd = optimizer.state_dict()

        # Check that the optimizer state has been updated
        assert initial_optimizer_sd != post_loss_optimizer_sd

        # Check that optimizer.step() doesn't update anything
        optimizer.step()
        assert optimizer.state_dict() == post_loss_optimizer_sd

    def test_adamw_with_scheduler(self, dummy_model, dummy_input_and_target):
        """Test that it works when paired with a learning rate scheduler."""
        initial_lr = 0.1
        optimizer = OptimizerInBackward(
            dummy_model.parameters(),
            AdamW,
            lr=initial_lr,
        )
        # Assert that initial_lr has NOT been set pre-LR scheduler init
        for param_group in optimizer.param_groups:
            assert "initial_lr" not in param_group

        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        # Assert that initial_lr has ben assigned to optimizer param groups
        for param_group in optimizer.param_groups:
            assert param_group["initial_lr"] == initial_lr

        optimizer.zero_grad()
        output = dummy_model(dummy_input_and_target[0])
        loss = nn.functional.mse_loss(output, dummy_input_and_target[1])
        loss.backward()

        optimizer.step()  # Do this so lr scheduler doesn't complain
        scheduler.step()
        post_step_lr = scheduler.get_last_lr()[0]
        assert post_step_lr == initial_lr * 0.1  # lr should be multiplied by gamma

    def test_throws_error_when_step_called_with_closure(self, dummy_model):
        """We don't implement with closure yet."""
        optimizer = OptimizerInBackward(
            dummy_model.parameters(),
            AdamW,
            lr=0.1,
        )
        with pytest.raises(RuntimeError):
            optimizer.step(lambda: None)

    def test_state_dict_save_load(self, dummy_model, dummy_input_and_target):
        optimizer = OptimizerInBackward(
            dummy_model.parameters(),
            AdamW,
            lr=0.1,
        )
        output = dummy_model(dummy_input_and_target[0])
        loss = nn.functional.mse_loss(output, dummy_input_and_target[1])
        loss.backward()
        state_dict = optimizer.state_dict()
        # Initialize a new OptimizerInBackward but with the same parameters
        new_optimizer = OptimizerInBackward(
            dummy_model.parameters(),
            AdamW,
            lr=0.1,
        )
        # It shouldn't match the old optimizer state dict to start
        assert new_optimizer.state_dict() != state_dict
        new_optimizer.load_state_dict(state_dict)
        # But after loading, it should match
        assert new_optimizer.state_dict() == state_dict

    def test_state_dict_save_load_with_scheduler(
        self, dummy_model, dummy_input_and_target
    ):
        optimizer = OptimizerInBackward(
            dummy_model.parameters(),
            AdamW,
            lr=0.1,
        )
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        output = dummy_model(dummy_input_and_target[0])
        loss = nn.functional.mse_loss(output, dummy_input_and_target[1])
        loss.backward()
        optimizer.step()  # Do this so lr scheduler doesn't complain
        scheduler.step()

        for param_group in optimizer.param_groups:
            assert param_group["initial_lr"] == 0.1

        opt_state_dict = optimizer.state_dict()

        # Initialize a new OptimizerInBackward but with the same parameters
        new_optimizer = OptimizerInBackward(
            dummy_model.parameters(),
            AdamW,
            lr=0.1,
        )
        new_optimizer.load_state_dict(opt_state_dict)
        # Create a new LR scheduler with the new optimizer
        new_scheduler = StepLR(new_optimizer, step_size=1, gamma=0.1, last_epoch=0)

        # Now check that the last learning rates match
        scheduler.step()  # <- THIS IS BAD, BUT NEEDED UNTIL https://github.com/pytorch/pytorch/pull/149312 LANDS
        assert scheduler.get_last_lr() == new_scheduler.get_last_lr()
