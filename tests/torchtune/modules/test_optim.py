# test_fused_optimizer_in_backward.py
import pytest

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchtune.modules.optim import OptimizerInBackward


class TestOptimizerInBackward:

    def dummy_model(self):
        return nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
        )

    def test_adamw(self, dummy_model):
        model = dummy_model()
        optimizer = OptimizerInBackward(
            model.parameters(),
            AdamW,
            lr=0.1,
        )
        optimizer.step()

    def test_adamw_with_scheduler(self, dummy_model):
        model = dummy_model()
        optimizer = OptimizerInBackward(
            model.parameters(),
            AdamW,
            lr=0.1,
        )
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        optimizer.step()
        scheduler.step()

    def test_saves_memory(self, dummy_model):
        model = dummy_model()
        optimizer = OptimizerInBackward(
            model.parameters(),
            AdamW,
            lr=0.1,
        )
        optimizer.step()
        # Check that the optimizer does not hold on to the model parameters
        assert len(optimizer.state) == 0

    def test_throws_error_when_step_called_with_closure(self, dummy_model):
        model = dummy_model()
        optimizer = OptimizerInBackward(
            model.parameters(),
            AdamW,
            lr=0.1,
        )
        with pytest.raises(RuntimeError):
            optimizer.step(lambda: None)

    def test_state_dict_save_load(self, dummy_model):
        model = dummy_model()
        optimizer = OptimizerInBackward(
            model.parameters(),
            AdamW,
            lr=0.1,
        )
        optimizer.step()
        state_dict = optimizer.state_dict()
        optimizer.load_state_dict(state_dict)

    def test_state_dict_save_load_with_scheduler(self, dummy_model):
        model = dummy_model()
        optimizer = OptimizerInBackward(
            model.parameters(),
            AdamW,
            lr=0.1,
        )
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        optimizer.step()
        scheduler.step()

        # Save and load optimizer state dict
        optimizer_state_dict = optimizer.state_dict()
        optimizer.load_state_dict(optimizer_state_dict)

        # Save and load scheduler state dict separately
        scheduler_state_dict = scheduler.state_dict()
        scheduler.load_state_dict(scheduler_state_dict)
