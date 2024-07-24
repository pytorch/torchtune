# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtune.modules.loss import SimPOLoss


@pytest.fixture(autouse=True)
def random():
    torch.manual_seed(16)


class TestDPOLosses:
    @pytest.fixture
    def simpo_loss(self):
        return SimPOLoss(
            beta=2.0,
            gamma=0.5,
            label_smoothing=0.0,
        )

    @pytest.fixture
    def loss_inputs(self):
        """
        kind-of-random inputs for testing the math out (below).
        """
        policy_chosen_logprobs = torch.tensor([-0.5, -10.0, -1.0])
        policy_rejected_logprobs = torch.tensor([-0.1, -30.0, -21.0])

        return (
            policy_chosen_logprobs,
            policy_rejected_logprobs,
        )

    def test_simpo_loss(self, simpo_loss, loss_inputs):
        """
        here's the maths (see `loss_inputs`):
        ratios = torch.tensor([-0.4, 20.0, 20.0])
        gamma_logratios = 0.25

            logits is ratios - gamma_logratios

        logits = torch.tensor([-0.65, 19.75, 19.75])
        scaled_logits = beta * logits = torch.tensor([-1.3,  39.5, 39.5])

        since label_smoothing is zero, loss is NLL with temperature scaled logits
        """
        exp_scaled_logits = torch.exp(torch.tensor([1.3, -39.5, -39.5]))

        expected_losses = -(1 / (1 + exp_scaled_logits)).log()
        losses, *_ = simpo_loss(*loss_inputs)

        torch.testing.assert_close(losses, expected_losses, atol=1e-4, rtol=1e-5)
