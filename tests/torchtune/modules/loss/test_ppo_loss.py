# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtune.modules.loss import PPOLoss


@pytest.fixture(autouse=True)
def random():
    torch.manual_seed(16)


class TestPPOLoss:
    @pytest.fixture
    def loss_fn(self):
        return PPOLoss(
            gamma=0.99,
            lmbda=0.95,
            value_clip_range=0.2,
            value_coeff=0.1,
            epsilon=1e-5,
        )

    def test_policy_loss_clipped_for_high_logprobs(self, loss_fn):
        # fixed old policy logprobs, advantages, returns
        pi_old_logprobs = torch.tensor([0.5, 0.8, 1.2])
        advantages = torch.tensor([1.0, 1.0, 1.0])
        values = torch.tensor([1.0, 1.0, 1.0])
        returns = torch.tensor([1.0, 1.0, 1.0])

        pi_logprobs_high = torch.tensor([0.5, 0.8, 1.2])

        _, policy_loss, _ = loss_fn(
            pi_old_logprobs, pi_logprobs_high, advantages, values, returns
        )

        # ratio will be clipped between 1 - epsilon, 1 + epsilon
        # ppo objective becomes advantages * (1 - epsilon) or  advantages * (1 + epsilon),
        # which is < advantages * unclipped ratios
        # policy loss is max (-ppo objective, -advantages * unclipped ratios)
        expected_loss = -advantages.mean()

        torch.testing.assert_close(
            policy_loss.mean(), expected_loss, atol=loss_fn.epsilon, rtol=0.0
        )

    def test_policy_loss_lower_for_higher_advantages(self, loss_fn):
        pi_logprobs = torch.tensor([-0.5, -0.8, -1.2])

        advantages_high = torch.tensor([1.0, 2.0, 3.0])
        advantages_low = torch.tensor([0.5, 1.0, 1.5])
        values = torch.tensor([1.0, 1.0, 1.0])
        returns = torch.tensor([1.0, 1.0, 1.0])

        _, policy_loss_low, _ = loss_fn(
            pi_logprobs, pi_logprobs, advantages_high, values, returns
        )
        _, policy_loss_high, _ = loss_fn(
            pi_logprobs, pi_logprobs, advantages_low, values, returns
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

        _, _, value_loss_lower = loss_fn(
            pi_logprobs, pi_logprobs, advantages, values_similar, returns
        )
        _, _, value_loss_higher = loss_fn(
            pi_logprobs, pi_logprobs, advantages, values_less_similar, returns
        )
        assert value_loss_lower.mean() < value_loss_higher.mean()
