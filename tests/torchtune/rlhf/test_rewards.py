# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtune import rlhf


class TestGetRewards:
    def test_get_rewards(self):
        scores = torch.tensor([1.0, 2.0, 3.0])
        logprobs = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.6, 0.7, 0.8],
            ]
        )
        ref_logprobs = torch.tensor(
            [
                [0.2, 0.3, 0.4],
                [0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1],
            ]
        )
        kl_controller_value = 0.5

        # expected kl is logprobs - ref_logprobs
        expected_kl = torch.tensor(
            [
                [-0.1, -0.1, -0.1],
                [-0.2, -0.2, -0.2],
                [-0.3, -0.3, -0.3],
            ]
        )

        # expected kl_rewards is -kl_controller_value * kl
        expected_kl_rewards = torch.tensor(
            [
                [0.05, 0.05, 0.05],
                [0.1, 0.1, 0.1],
                [0.15, 0.15, 0.15],
            ]
        )

        # expected rewards is kl_rewards[:, -1] + scores
        expected_rewards = torch.tensor(
            [
                [0.05, 0.05, 1.05],
                [0.1, 0.1, 2.1],
                [0.15, 0.15, 3.15],
            ]
        )

        rewards, kl, kl_rewards = rlhf.get_rewards_ppo(
            scores, logprobs, ref_logprobs, kl_controller_value
        )

        torch.testing.assert_close(kl, expected_kl, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            kl_rewards, expected_kl_rewards, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(rewards, expected_rewards, rtol=1e-4, atol=1e-4)


class TestWhiten:
    def test_whiten_with_shift_mean(self):
        x = torch.normal(1, 2, size=(100, 100))

        expected_mean, expected_var = x.mean(), x.var()  # should be ~1.0, ~4.0
        expected = (x - expected_mean) / (torch.sqrt(expected_var) + 1e-8)
        expected += expected_mean
        output = rlhf.whiten(x, shift_mean=True)

        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_whiten_without_shift_mean(self):
        x = torch.normal(1, 2, size=(100, 100))

        expected_mean, expected_var = x.mean(), x.var()  # should be ~1.0, ~4.0
        expected = (x - expected_mean) / (torch.sqrt(expected_var) + 1e-8)
        output = rlhf.whiten(x, shift_mean=False)

        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_masked_whiten(self):
        x_mean_1 = torch.normal(1, 2, size=(50, 100))
        x_mean_2 = torch.normal(2, 1, size=(50, 100))
        x = torch.cat([x_mean_1, x_mean_2], dim=0)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:50] = False

        expected_mean, expected_var = (
            x_mean_2.mean(),
            x_mean_2.var(),
        )  # should be ~2.0, ~1.0
        expected = (x - expected_mean) / (torch.sqrt(expected_var) + 1e-8)
        expected += expected_mean

        output = rlhf.whiten(x, mask=mask)

        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


class TestMaskedMean:
    def test_masked_single_batch_mean(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = torch.tensor([True, True, True, False, False])

        expected_mean = torch.tensor(2.0)
        output = rlhf.masked_mean(x, mask)

        torch.testing.assert_close(output, expected_mean, rtol=1e-4, atol=1e-4)

    def test_masked_multi_batch_mean(self):
        x = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
            ]
        )
        mask = torch.tensor(
            [[True, True, True, False, False], [False, False, True, True, True]]
        )

        expected_means = torch.tensor([2.0, 5.0])
        output = rlhf.masked_mean(x, mask, dim=1)

        torch.testing.assert_close(output, expected_means, rtol=1e-4, atol=1e-4)


class TestMaskedVar:
    def test_masked_var(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = torch.tensor([True, True, True, False, False])

        expected_var = torch.tensor(1.0)
        output = rlhf.masked_var(x, mask)

        torch.testing.assert_close(output, expected_var, rtol=1e-4, atol=1e-4)


class TestEstimateAdvantages:
    def test_estimate_returns(self):
        values = torch.tensor([[0, 0, 0, 1]])
        rewards = torch.tensor([[0, 0, 0, 1]])
        gamma = 0.9
        lmbda = 0.95

        final_reward = 1.0
        expected_returns = torch.tensor(
            [
                [
                    final_reward * gamma * gamma * gamma * lmbda * lmbda,
                    final_reward * gamma * gamma * lmbda,
                    final_reward * gamma,
                    final_reward,
                ]
            ]
        )

        _, returns = rlhf.estimate_advantages(values, rewards, gamma, lmbda)
        torch.testing.assert_close(returns, expected_returns, rtol=1e-4, atol=1e-4)

    def test_estimate_advantages_with_whitening(self):
        values = torch.tensor([[0, 0, 0, 1]])
        rewards = torch.tensor([[0, 0, 0, 1]])
        gamma = 0.9
        lmbda = 0.95

        final_reward = 1.0
        returns = torch.tensor(
            [
                [
                    final_reward * gamma * gamma * gamma * lmbda * lmbda,
                    final_reward * gamma * gamma * lmbda,
                    final_reward * gamma,
                    final_reward,
                ]
            ]
        )

        # see `torchtune.rlhf.estimate_advantages`
        expected_advantages = returns - values
        expected_whitened_advantages = rlhf.whiten(expected_advantages, shift_mean=True)
        advantages, _ = rlhf.estimate_advantages(values, rewards, gamma, lmbda)
        torch.testing.assert_close(
            expected_whitened_advantages, advantages, rtol=1e-4, atol=1e-4
        )

    def test_estimate_advantages_with_masks(self):
        values = torch.tensor([[0, 0, 0, 1]])
        rewards = torch.tensor([[0, 0, 0, 1]])
        masks = torch.tensor([[True, True, True, False]])
        gamma = 0.9
        lmbda = 0.95

        final_reward = 1.0
        returns = torch.tensor(
            [
                [
                    final_reward * gamma * gamma * gamma * lmbda * lmbda,
                    final_reward * gamma * gamma * lmbda,
                    final_reward * gamma,
                    final_reward,
                ]
            ]
        )

        # see `torchtune.rlhf.estimate_advantages`
        expected_advantages = returns - values
        expected_advantages = rlhf.whiten(expected_advantages, mask=masks)
        expected_advantages[..., -1] = 0.0

        advantages, _ = rlhf.estimate_advantages(
            values, rewards, gamma, lmbda, masks=masks
        )
        torch.testing.assert_close(
            advantages, expected_advantages, rtol=1e-4, atol=1e-4
        )
