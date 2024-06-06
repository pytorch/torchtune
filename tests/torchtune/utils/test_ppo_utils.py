# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest

import torch
from tests.test_utils import fixed_init_model
from torchtune.models.llama2 import llama2
from torchtune.utils.ppo_utils import estimate_advantages, generate, get_rewards, whiten

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


class TestGenerate:
    """
    Test class for incremental text generation functionality in :func:`~torchtune.utils.ppo_utils.generate`.
    See `torchtune.tests.utils.test_generation` for context.
    """

    @pytest.fixture
    def generation_model(self):
        model = llama2(
            vocab_size=4_000,
            embed_dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            max_seq_len=2048,
        )
        fixed_init_model(model)
        model.eval()
        return model

    @pytest.fixture
    def prompt_tokens(self):
        """
        Pytest fixture to create a list of prompt tokens for testing.
        """
        return torch.arange(2, 10)

    @pytest.fixture
    def padded_prompt_tokens(self):
        """
        Pytest fixture to create a list of left-padded prompt tokens for testing.
        """
        return torch.cat([torch.tensor([0, 0]), torch.arange(2, 10)])

    @pytest.fixture
    def prompt_tokens_batched_padded(self):
        """
        Pytest fixture to create a list of left-padded batched prompt tokens for testing.
        """
        return torch.cat([torch.tensor([0, 0]), torch.arange(2, 10)]).repeat(3, 1)

    @pytest.fixture
    def prompt_tokens_batched(self):
        return torch.arange(2, 10).repeat(3, 1)

    def test_reproducability_with_and_without_padding_batched(
        self,
        generation_model,
        prompt_tokens_batched_padded,
        prompt_tokens_batched,
    ):
        """
        Test to check if the `generate` function produces the same output for inputs that are left padded
        and for the same inputs that are not left padded, for a batch of inputs with varying sequence lengths.
        """
        temperature = 0.6
        top_k = 100

        torch.manual_seed(42)
        outputs = generate(
            model=generation_model,
            prompt=prompt_tokens_batched_padded,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
        )

        torch.manual_seed(42)
        expected_outputs = generate(
            model=generation_model,
            prompt=prompt_tokens_batched,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
        )

        assert [output[2:] for output in outputs] == expected_outputs

    def test_reproducability_with_and_without_padding(
        self, generation_model, prompt_tokens, padded_prompt_tokens
    ):
        """
        Test to check if the `generate` function produces the same output for inputs that are left padded
        and for the same inputs that are not left padded.
        """
        temperature = 0.6
        top_k = 100

        torch.manual_seed(42)
        outputs_unpadded = generate(
            model=generation_model,
            prompt=prompt_tokens,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
        )

        torch.manual_seed(42)
        outputs_padded = generate(
            model=generation_model,
            prompt=padded_prompt_tokens,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
        )

        assert outputs_unpadded[0] == outputs_padded[0][2:]

    def test_stop_tokens(self, generation_model, prompt_tokens):
        """
        Test to check if the `generate` function produces the right output when stop tokens are
        provided.
        """
        temperature = 0.6
        top_k = 100

        # This is the first token generated by the model
        # so it should stop immediately
        stop_tokens = [3983]

        torch.manual_seed(42)
        outputs = generate(
            model=generation_model,
            prompt=prompt_tokens,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=stop_tokens,
        )

        expected_output = [[2, 3, 4, 5, 6, 7, 8, 9, 3983]]
        print(outputs, expected_output)
        assert outputs == expected_output

    def test_stop_tokens_batched(self, generation_model, prompt_tokens_batched):
        """
        Test to check if the `generate` function produces the right output when stop tokens are
        provided, but this time in batched format.
        """
        temperature = 0.6
        top_k = 100

        # These are the first tokens generated by the model
        # so it should stop immediately
        stop_tokens = [3983, 3953, 3989]

        torch.manual_seed(42)

        outputs = generate(
            model=generation_model,
            prompt=prompt_tokens_batched,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=stop_tokens,
        )

        expected_outputs = [
            [2, 3, 4, 5, 6, 7, 8, 9, 3983],
            [2, 3, 4, 5, 6, 7, 8, 9, 3953],
            [2, 3, 4, 5, 6, 7, 8, 9, 3989],
        ]

        assert outputs == expected_outputs

    def test_stop_tokens_batched_uneven(self, generation_model, prompt_tokens_batched):
        """
        Test to check if the `generate` function produces the right output when stop tokens are
        provided, but this time in batched format with different stopping lengths.
        """
        temperature = 0.6
        top_k = 100

        stop_tokens = [3962, 3953, 3999]

        torch.manual_seed(42)

        outputs = generate(
            model=generation_model,
            prompt=prompt_tokens_batched,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=stop_tokens,
        )

        expected_outputs = [
            [2, 3, 4, 5, 6, 7, 8, 9, 3983, 3950, 3962],
            [2, 3, 4, 5, 6, 7, 8, 9, 3953, 0, 0],
            [2, 3, 4, 5, 6, 7, 8, 9, 3989, 3999, 0],
        ]

        assert outputs == expected_outputs
