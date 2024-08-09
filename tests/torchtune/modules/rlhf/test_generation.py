# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import fixed_init_model
from torchtune.models.llama2 import llama2
from torchtune.modules import rlhf
from torchtune.utils._generation import sample


class TestGenerateNextTokenWithLogits:
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

    def test_generate_next_token_with_logits(self, generation_model):

        inputs = torch.tensor(
            [
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11],
            ]
        )

        input_pos = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
            ]
        )

        torch.manual_seed(42)
        logits, generation = rlhf.generate_next_token_with_logits(
            generation_model, input_pos, inputs
        )

        torch.manual_seed(42)
        expected_logits = generation_model(inputs, input_pos=input_pos)
        expected_generation = sample(logits[:, -1], temperature=1.0, top_k=None)

        torch.testing.assert_close(logits, expected_logits, atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(generation, expected_generation, atol=0, rtol=0)


class TestGenerate:
    """
    Test class for text generation functionality in :func:`~torchtune.modules.rlhf.generate`.
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
    def prompt_tokens_batched(self):
        """
        Pytest fixture to create a list of batched prompt tokens for testing.
        """
        return torch.arange(2, 10).repeat(3, 1)

    @pytest.fixture
    def prompt_tokens_padded(self):
        """
        Pytest fixture to create a list of left-padded prompt tokens for testing.
        """
        return torch.cat([torch.tensor([0, 0]), torch.arange(2, 10)])

    @pytest.fixture
    def prompt_tokens_batched_left_padded(self):
        """
        Pytest fixture to create a list of left-padded batched prompt tokens for testing.
        """
        return torch.cat([torch.tensor([0, 0]), torch.arange(2, 10)]).repeat(3, 1)

    def test_reproducability_with_and_without_padding_batched(
        self,
        generation_model,
        prompt_tokens_batched_left_padded,
        prompt_tokens_batched,
    ):
        """
        Test to check if the `generate` function produces the same output for inputs that are left padded
        and for the same inputs that are not left padded, for a batch of inputs with varying sequence lengths.
        """
        temperature = 0.6
        top_k = 100

        torch.manual_seed(42)
        outputs, _ = rlhf.generate_with_logits(
            model=generation_model,
            prompt=prompt_tokens_batched_left_padded,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
        )

        torch.manual_seed(42)
        expected_outputs, _ = rlhf.generate_with_logits(
            model=generation_model,
            prompt=prompt_tokens_batched,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
        )

        torch.testing.assert_close(outputs[:, 2:], expected_outputs, atol=0, rtol=0)

    def test_reproducability_with_and_without_padding(
        self, generation_model, prompt_tokens, prompt_tokens_padded
    ):
        """
        Test to check if the `generate` function produces the same output for inputs that are left padded
        and for the same inputs that are not left padded.
        """
        temperature = 0.6
        top_k = 100

        torch.manual_seed(42)

        outputs, _ = rlhf.generate_with_logits(
            model=generation_model,
            prompt=prompt_tokens_padded,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
        )

        torch.manual_seed(42)
        expected_outputs, _ = rlhf.generate_with_logits(
            model=generation_model,
            prompt=prompt_tokens,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
        )

        torch.testing.assert_close(outputs[:, 2:], expected_outputs, atol=0, rtol=0)


class TestGetCausalMask:
    @pytest.fixture
    def left_padded_prompt_tokens(self):
        """
        Pytest fixture to create a list of left-padded prompt tokens for testing.
        """
        return torch.cat([torch.tensor([0, 0]), torch.arange(2, 6)]).unsqueeze(0)

    @pytest.fixture
    def left_padded_prompt_tokens_batched(self):
        """
        Pytest fixture to create a list of left-padded batched prompt tokens for testing.
        """
        return torch.tensor(
            [[0, 0, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 1]]
        )

    @pytest.fixture
    def right_padded_prompt_tokens(self):
        """
        Pytest fixture to create a list of right-padded prompt tokens for testing.
        """
        return torch.cat([torch.arange(2, 6), torch.tensor([0, 0])]).unsqueeze(0)

    @pytest.fixture
    def right_padded_prompt_tokens_batched(self):
        """
        Pytest fixture to create a list of right-padded batched prompt tokens for testing.
        """
        return torch.tensor(
            [[1, 2, 3, 4, 5, 0], [1, 2, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6]]
        )

    @pytest.fixture
    def mixed_padded_prompt_tokens(self):
        """
        Pytest fixture to create a list of mixed padded prompt tokens for testing.
        """
        return torch.cat(
            [torch.tensor([0, 0]), torch.arange(2, 6), torch.tensor([0, 0])]
        ).unsqueeze(0)

    @pytest.fixture
    def mixed_padded_prompt_tokens_batched(self):
        """
        Pytest fixture to create a list of mixed padded batched prompt tokens for testing.
        """
        return torch.tensor(
            [[0, 0, 1, 2, 0, 0], [0, 1, 2, 3, 4, 0], [0, 0, 0, 1, 0, 0]]
        )

    def test_get_causal_mask_for_left_padded_inputs(self, left_padded_prompt_tokens):
        """
        Test to check if the `get_causal_mask` function produces the right output for left-padded prompts.
        """
        expected_casual_mask = torch.tensor(
            [
                [True, False, False, False, False, False],
                [False, True, False, False, False, False],
                [False, False, True, False, False, False],
                [False, False, True, True, False, False],
                [False, False, True, True, True, False],
                [False, False, True, True, True, True],
            ]
        ).unsqueeze(0)

        causal_mask = rlhf.get_causal_mask(left_padded_prompt_tokens != 0)
        torch.testing.assert_close(causal_mask, expected_casual_mask, atol=0, rtol=0)

    def test_get_causal_mask_for_left_padded_inputs_batched(
        self, left_padded_prompt_tokens_batched
    ):
        """
        Test to check if the `get_causal_mask` function produces the right output for left-padded batched prompts.
        """
        expected_causal_mask = torch.tensor(
            [
                [
                    [True, False, False, False, False, False],
                    [False, True, False, False, False, False],
                    [False, False, True, False, False, False],
                    [False, False, False, True, False, False],
                    [False, False, False, True, True, False],
                    [False, False, False, True, True, True],
                ],
                [
                    [True, False, False, False, False, False],
                    [False, True, False, False, False, False],
                    [False, True, True, False, False, False],
                    [False, True, True, True, False, False],
                    [False, True, True, True, True, False],
                    [False, True, True, True, True, True],
                ],
                [
                    [True, False, False, False, False, False],
                    [False, True, False, False, False, False],
                    [False, False, True, False, False, False],
                    [False, False, False, True, False, False],
                    [False, False, False, False, True, False],
                    [False, False, False, False, False, True],
                ],
            ]
        )

        causal_mask = rlhf.get_causal_mask(left_padded_prompt_tokens_batched != 0)
        torch.testing.assert_close(causal_mask, expected_causal_mask, atol=0, rtol=0)

    def test_get_causal_mask_for_right_padded_inputs(self, right_padded_prompt_tokens):
        """
        Test to check if the `get_causal_mask` function produces the right output for right-padded prompts.
        """
        expected_causal_mask = torch.tensor(
            [
                [True, False, False, False, False, False],
                [True, True, False, False, False, False],
                [True, True, True, False, False, False],
                [True, True, True, True, False, False],
                [False, False, False, False, True, False],
                [False, False, False, False, False, True],
            ]
        ).unsqueeze(0)

        causal_mask = rlhf.get_causal_mask(right_padded_prompt_tokens != 0)
        torch.testing.assert_close(causal_mask, expected_causal_mask, atol=0, rtol=0)

    def test_get_causal_mask_for_right_padded_inputs_batched(
        self, right_padded_prompt_tokens_batched
    ):
        """
        Test to check if the `get_causal_mask` function produces the right output for right-padded batched prompts.
        """
        expected_causal_mask = torch.tensor(
            [
                [
                    [True, False, False, False, False, False],
                    [True, True, False, False, False, False],
                    [True, True, True, False, False, False],
                    [True, True, True, True, False, False],
                    [True, True, True, True, True, False],
                    [False, False, False, False, False, True],
                ],
                [
                    [True, False, False, False, False, False],
                    [True, True, False, False, False, False],
                    [False, False, True, False, False, False],
                    [False, False, False, True, False, False],
                    [False, False, False, False, True, False],
                    [False, False, False, False, False, True],
                ],
                [
                    [True, False, False, False, False, False],
                    [True, True, False, False, False, False],
                    [True, True, True, False, False, False],
                    [True, True, True, True, False, False],
                    [True, True, True, True, True, False],
                    [True, True, True, True, True, True],
                ],
            ]
        )

        causal_mask = rlhf.get_causal_mask(right_padded_prompt_tokens_batched != 0)
        torch.testing.assert_close(causal_mask, expected_causal_mask, atol=0, rtol=0)

    def test_get_causal_mask_for_mixed_padding_inputs(self, mixed_padded_prompt_tokens):
        """
        Test to check if the `get_causal_mask` function produces the right output for mixed padded prompts.
        """
        expected_causal_mask = torch.tensor(
            [
                [True, False, False, False, False, False, False, False],
                [False, True, False, False, False, False, False, False],
                [False, False, True, False, False, False, False, False],
                [False, False, True, True, False, False, False, False],
                [False, False, True, True, True, False, False, False],
                [False, False, True, True, True, True, False, False],
                [False, False, False, False, False, False, True, False],
                [False, False, False, False, False, False, False, True],
            ]
        ).unsqueeze(0)

        causal_mask = rlhf.get_causal_mask(mixed_padded_prompt_tokens != 0)
        torch.testing.assert_close(causal_mask, expected_causal_mask, atol=0, rtol=0)

    def test_get_causal_mask_for_mixed_padded_inputs_batched(
        self, mixed_padded_prompt_tokens_batched
    ):
        """
        Test to check if the `get_causal_mask` function produces the right output for mixed-padded batched prompts.
        """
        expected_causal_mask = torch.tensor(
            [
                [
                    [True, False, False, False, False, False],
                    [False, True, False, False, False, False],
                    [False, False, True, False, False, False],
                    [False, False, True, True, False, False],
                    [False, False, False, False, True, False],
                    [False, False, False, False, False, True],
                ],
                [
                    [True, False, False, False, False, False],
                    [False, True, False, False, False, False],
                    [False, True, True, False, False, False],
                    [False, True, True, True, False, False],
                    [False, True, True, True, True, False],
                    [False, False, False, False, False, True],
                ],
                [
                    [True, False, False, False, False, False],
                    [False, True, False, False, False, False],
                    [False, False, True, False, False, False],
                    [False, False, False, True, False, False],
                    [False, False, False, False, True, False],
                    [False, False, False, False, False, True],
                ],
            ]
        )

        causal_mask = rlhf.get_causal_mask(mixed_padded_prompt_tokens_batched != 0)
        torch.testing.assert_close(causal_mask, expected_causal_mask, atol=0, rtol=0)
