# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtune.rlhf.loss import DPOLoss, KTOLoss, RSOLoss, SimPOLoss


@pytest.fixture(autouse=True)
def random():
    torch.manual_seed(16)


class TestDPOLosses:
    @pytest.fixture
    def dpo_loss(self):
        return DPOLoss(
            beta=0.1,
            label_smoothing=0.0,
        )

    @pytest.fixture
    def rso_loss(self):
        return RSOLoss(
            gamma=0.1,
        )

    @pytest.fixture
    def simpo_loss(self):
        return SimPOLoss(
            beta=2.0,
            gamma=0.5,
            label_smoothing=0.0,
        )

    @pytest.fixture
    def kto_loss(self):
        return KTOLoss(
            beta=0.1,
            undesirable_weight=1.0,
            desirable_weight=1.0,
        )

    @pytest.fixture
    def loss_inputs(self):
        """
        kind-of-random inputs for testing the math out (below).
        """
        policy_chosen_logprobs = torch.tensor([-0.5, -10.0, -1.0])
        policy_rejected_logprobs = torch.tensor([-0.1, -30.0, -21.0])

        ref_chosen_logprobs = torch.tensor([-0.5, -10.1, -0.1])
        ref_rejected_logprobs = torch.tensor([-0.1, -20.1, -0.1])



        return (
            policy_chosen_logprobs,
            policy_rejected_logprobs,
            ref_chosen_logprobs,
            ref_rejected_logprobs,
        )

    def test_dpo_loss(self, dpo_loss, loss_inputs):
        """
        here's the maths (see `loss_inputs`):
        ratios = torch.tensor([-0.4, 20.0, 20.0])
        ref_ratios = torch.tensor([-0.4, 10, 0.0])

            logits is ratios - ref_ratios

        logits = torch.tensor([0.0, 10.0, 20.0])
        scaled_logits = torch.tensor([0.0, 1.0, 2.0])

        since label_smoothing is zero, loss is NLL with temperature scaled logits
            logsigmoid is log(1/1+exp(-scaled_logits))
            exp(-scaled_logits) is [1, 1/e, 1/e^2]
            logsigmoid is -log([1 / 2, 1 / (1 + 1/e), 1 / (1 + 1/e^2)])

        expected_losses = -torch.tensor(
            [1 / 2, 1 / (1 + torch.exp(torch.tensor(-1.0))), 1 / (1 + torch.exp(torch.tensor(-2.0)))]
        ).log()
        expected_losses = -expected_logsigmoids
        """
        exp_scaled_logits = torch.exp(torch.tensor([0.0, -1.0, -2.0]))
        expected_losses = -(1 / (1 + exp_scaled_logits)).log()
        losses, *_ = dpo_loss(*loss_inputs)

        torch.testing.assert_close(losses, expected_losses, atol=1e-4, rtol=1e-5)

    def test_rso_loss(self, rso_loss, loss_inputs):
        """
        # maths:
        ratios = torch.tensor([-0.4, 20.0, 20.0])
        ref_ratios = torch.tensor([-0.4, 10, 0.0])

        # logits is ratios - ref_ratios

        logits = torch.tensor([0.0, 10.0, 20.0])
        scaled_logits = torch.tensor([0.0, 1.0, 2.0])

        # hinge loss doesn't use label smoothing
        # loss = relu(1 - scaled_logits) = max(0, 1 - scaled_logits)
        expected_losses = torch.tensor([1.0, 0.0, 0.0])
        """

        expected_losses = torch.tensor([1.0, 0.0, 0.0])

        losses, *_ = rso_loss(*loss_inputs)

        torch.testing.assert_close(losses, expected_losses, atol=1e-4, rtol=1e-5)

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
        policy_chosen_logprobs, policy_rejected_logprobs, *_ = loss_inputs
        exp_scaled_logits = torch.exp(torch.tensor([1.3, -39.5, -39.5]))

        expected_losses = -(1 / (1 + exp_scaled_logits)).log()
        losses, *_ = simpo_loss(policy_chosen_logprobs, policy_rejected_logprobs)

        torch.testing.assert_close(losses, expected_losses, atol=1e-4, rtol=1e-5)

    def test_kto_loss(self, kto_loss, loss_inputs):
        """
        beta = 0.1
        policy_chosen_logprobs = torch.tensor([-0.5, -10.0, -1.0])
        policy_rejected_logprobs = torch.tensor([-0.1, -30.0, -21.0])

        ref_chosen_logprobs = torch.tensor([-0.5, -10.1, -0.1])
        ref_rejected_logprobs = torch.tensor([-0.1, -20.1, -0.1])

        policy_KL_logps = torch.tensor([-1.0, 0.25, 1.0])
        reference_KL_logps = torch.tensor([0.4, 0.2, -0.2])

        kl = (policy_KL_logps - reference_KL_logps).mean().detach()
        kl = kl.mean().clamp(min=0)

        kl = torch.tensor([0])

        chosen_logratios = policy_chosen_logprobs - ref_chosen_logprobs
        chosen_logratios = torch.tensor([0., 0.1, -0.9])

        chosen_losses = 1 - F.sigmoid(0.1 * (torch.tensor([0., 0.1, -0.9]) - torch.tensor([0])))
        chosen_losses = torch.tensor([0.5000, 0.4975, 0.5225])

        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        rejected_logratios = torch.tensor([0.0,  -9.9, -20.9])

        rejected_losses = 1 - F.sigmoid(0.1 * (torch.tensor([0]) - torch.tensor([0.0,  -9.9, -20.9])))
        rejected_losses = torch.tensor([0.5, 0.2709, 0.1101])


        desirable_weight = undesirable_weight = 1.0
        Therefore:

        losses = torch.cat(
             (1 * chosen_losses, 1 * rejected_losses),
             0
        )

        losses = torch.tensor([0.5000, 0.4975, 0.5225, 0.5000, 0.2709, 0.1101])

        chosen_rewards = 0.1 * (chosen_logratios).detach()
        rejected_rewards = 0.1 * (rejected_logratios).detach()

        chosen_rewards = torch.tensor([0.0000,  0.0100, -0.0900])
        rejected_rewards = torch.tensor([ 0.0000, -0.9900, -2.0900])
        """

        (
            policy_chosen_logprobs,
            policy_rejected_logprobs,
            ref_chosen_logprobs,
            ref_rejected_logprobs,
        ) = loss_inputs

        # We don't add it in general fixture, as it is only relevant for KTO
        policy_KL_logps = torch.tensor([-1.0, 0.25, 1.0])
        reference_KL_logps = torch.tensor([0.4, 0.2, -0.2])

        losses, *_ = kto_loss(
            policy_chosen_logprobs,
            policy_rejected_logprobs,
            policy_KL_logps,
            ref_chosen_logprobs,
            ref_rejected_logprobs,
            reference_KL_logps
        )

        expected_losses = torch.tensor([0.5000, 0.4975, 0.5225, 0.5000, 0.2709, 0.1101])
        torch.testing.assert_close(losses, expected_losses, atol=1e-4, rtol=1e-5)
