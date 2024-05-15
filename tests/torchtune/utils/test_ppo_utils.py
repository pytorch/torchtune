# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchtune.utils.ppo_utils import estimate_advantages, get_rewards, whiten


# class TestAdaptiveKLController(unittest.TestCase):
#     def test_update(self):
#         # TODO (SalmanMohammadi)
#         # Test case 1: KL value is below target
#         controller = AdaptiveKLController(0.5, 1.0, 10)
#         controller.update(0.8, 5)
#         self.assertEqual(controller.value, 0.5)  # KL value should remain the same

#         # Test case 2: KL value is above target
#         controller = AdaptiveKLController(0.5, 1.0, 10)
#         controller.update(1.2, 5)
#         self.assertLess(controller.value, 0.5)  # KL value should decrease


class TestGetRewards(unittest.TestCase):
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

        rewards, kl, kl_rewards = get_rewards(
            scores, logprobs, ref_logprobs, kl_controller_value
        )

        torch.testing.assert_close(kl, expected_kl, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            kl_rewards, expected_kl_rewards, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(rewards, expected_rewards, rtol=1e-4, atol=1e-4)


class TestWhiten(unittest.TestCase):
    def test_whiten_with_shift_mean(self):
        x = torch.tensor([-1.0, 0.0, 4.0, 8.0, 3.0, -25.0])

        expected = torch.tensor([-1.7559, -1.6630, -1.2913, -0.9196, -1.3842, -3.9861])
        output = whiten(x, shift_mean=True)

        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_whiten_without_shift_mean(self):
        x = torch.tensor([-1.0, 0.0, 4.0, 8.0, 3.0, -25.0])

        expected = torch.tensor([0.0774, 0.1704, 0.5421, 0.9138, 0.4491, -2.1528])
        output = whiten(x, shift_mean=False)

        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


class TestEstimateAdvantages(unittest.TestCase):
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

        _, returns = estimate_advantages(values, rewards, gamma, lmbda)
        torch.testing.assert_close(returns, expected_returns, rtol=1e-4, atol=1e-4)

    def test_estimate_advantages(self):
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

        # see :func:`~torchtune.utils.ppo_utils.estimate_advantages`
        advantages = returns - values

        advantages, _ = estimate_advantages(values, rewards, gamma, lmbda)
        torch.testing.assert_close(advantages, advantages, rtol=1e-4, atol=1e-4)
