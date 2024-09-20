# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtune.rlhf.loss import PPOLoss


@pytest.fixture(autouse=True)
def random():
    torch.manual_seed(16)


class TestPPOLoss:
    @pytest.fixture
    def loss_fn(self):
        return PPOLoss(
            value_clip_range=0.2,
            value_coeff=0.1,
            epsilon=0.2,
        )

    def test_policy_loss_clipped_for_high_logprobs(self, loss_fn):
        # fixed old policy logprobs, advantages, returns
        pi_old_logprobs = torch.tensor([0.5, 0.8, 1.2])
        advantages = torch.tensor([1.0, 1.0, 1.0])
        values = torch.tensor([1.0, 1.0, 1.0])
        returns = torch.tensor([1.0, 1.0, 1.0])

        pi_logprobs_high = torch.tensor([1.5, 1.8, 2.2])
        # ratio will be [e, e, e]
        # clipped ratio becomes [1.2, 1.2, 1.2] (1+epsilon)
        # objective becomes max(-e, -1.2) since advantages is 1
        expected_loss = torch.tensor(-1.2)
        expected_ratios = torch.exp(torch.ones((3)))

        _, policy_loss, _, ratios, _ = loss_fn(
            pi_old_logprobs, pi_logprobs_high, advantages, values, values, returns
        )

        torch.testing.assert_close(
            policy_loss.mean(), expected_loss, atol=1e-4, rtol=1e6
        )
        torch.testing.assert_close(ratios, expected_ratios.mean(), atol=1e-4, rtol=1e6)

    def test_policy_loss_clipped_for_low_logprobs(self, loss_fn):
        # fixed old policy logprobs, advantages, returns
        pi_old_logprobs = torch.tensor([0.5, 0.8, 1.2])
        advantages = torch.tensor([1.0, 1.0, 1.0])
        values = torch.tensor([1.0, 1.0, 1.0])
        returns = torch.tensor([1.0, 1.0, 1.0])

        pi_logprobs_low = torch.tensor([-0.5, -0.2, 0.2])
        # ratio will be [1/e, 1/e, 1/e] (~0.367)
        # clipped ratio becomes [0.8, 0.8, 0.8] (1-epsilon)
        # objective becomes max(1/e, 0.8) since advantages is 1
        expected_loss = torch.tensor(0.8)
        expected_ratios = 1 / torch.exp(torch.ones((3)))

        _, policy_loss, _, ratios, _ = loss_fn(
            pi_old_logprobs, pi_logprobs_low, advantages, values, values, returns
        )

        torch.testing.assert_close(
            policy_loss.mean(), expected_loss, atol=1e-4, rtol=1e6
        )
        torch.testing.assert_close(ratios, expected_ratios.mean(), atol=1e-4, rtol=1e6)

    def test_policy_loss_not_clipped(self, loss_fn):
        # fixed old policy logprobs, advantages, returns
        pi_old_logprobs = torch.tensor([0.5, 0.8, 1.2])
        advantages = torch.tensor([1.0, 1.0, 1.0])
        values = torch.tensor([1.0, 1.0, 1.0])
        returns = torch.tensor([1.0, 1.0, 1.0])

        pi_logprobs_unclipped = torch.tensor([0.6, 0.9, 1.3])
        # ratio will be [e^0.1, e^0.1, e^0.1] (~1.1)
        # ratio is not clipped since it is within [1-epsilon, 1+epsilon], [0.8, 1.2]
        expected_loss = torch.tensor(0.1).exp()
        expected_ratios = torch.exp(torch.ones(3) * 0.1)

        _, policy_loss, _, ratios, _ = loss_fn(
            pi_old_logprobs, pi_logprobs_unclipped, advantages, values, values, returns
        )

        torch.testing.assert_close(
            policy_loss.mean(), expected_loss, atol=1e-4, rtol=1e6
        )
        torch.testing.assert_close(ratios, expected_ratios.mean(), atol=1e-4, rtol=1e6)

    def test_policy_loss_lower_for_higher_advantages(self, loss_fn):
        pi_logprobs = torch.tensor([-0.5, -0.8, -1.2])

        advantages_high = torch.tensor([1.0, 2.0, 3.0])
        advantages_low = torch.tensor([0.5, 1.0, 1.5])
        values = torch.tensor([1.0, 1.0, 1.0])
        returns = torch.tensor([1.0, 1.0, 1.0])

        _, policy_loss_low, *_ = loss_fn(
            pi_logprobs, pi_logprobs, advantages_high, values, values, returns
        )
        _, policy_loss_high, *_ = loss_fn(
            pi_logprobs, pi_logprobs, advantages_low, values, values, returns
        )

        assert policy_loss_low.mean() < policy_loss_high.mean()

    def test_value_loss_lower_for_values_similar_to_return(self, loss_fn):
        # fix pi_logrobs, pi_old_logprobs, returns, advantages
        pi_logprobs = torch.tensor([-0.5, -0.8, -1.2])
        returns = torch.tensor([1.0, 1.0, 1.0])
        advantages = torch.tensor([1.0, 1.0, 1.0])

        # values estimates are similar to returns
        values_similar = torch.tensor([0.9, 1.0, 1.1])
        # value estimates are less similar to returns
        values_less_similar = torch.tensor([0.5, 1.5, 2.0])

        _, _, value_loss_lower, *_ = loss_fn(
            pi_logprobs,
            pi_logprobs,
            advantages,
            values_similar,
            values_similar,
            returns,
        )
        _, _, value_loss_higher, *_ = loss_fn(
            pi_logprobs,
            pi_logprobs,
            advantages,
            values_similar,
            values_less_similar,
            returns,
        )
        assert value_loss_lower.mean() < value_loss_higher.mean()
