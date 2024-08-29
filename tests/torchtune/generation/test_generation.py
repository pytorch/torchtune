# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import fixed_init_model
from torchtune.generation import (
    generate,
    generate_next_token,
    get_causal_mask_from_padding_mask,
    get_position_ids_from_padding_masks,
)
from torchtune.generation._generation import sample
from torchtune.models.llama2 import llama2


class TestGenerate:
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

    def test_generate_next_token(self, generation_model):

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
        logits, generation = rlhf.generate_next_token(
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
        outputs, _ = generate(
            model=generation_model,
            prompt=prompt_tokens_batched_left_padded,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
        )

        torch.manual_seed(42)
        expected_outputs, _ = generate(
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

        outputs, _ = generate(
            model=generation_model,
            prompt=prompt_tokens_padded,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
        )

        torch.manual_seed(42)
        expected_outputs, _ = generate(
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

        causal_mask = rlhf.get_causal_mask_from_padding_mask(
            left_padded_prompt_tokens != 0
        )
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

        causal_mask = rlhf.get_causal_mask_from_padding_mask(
            left_padded_prompt_tokens_batched != 0
        )
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

        causal_mask = rlhf.get_causal_mask_from_padding_mask(
            right_padded_prompt_tokens != 0
        )
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

        causal_mask = rlhf.get_causal_mask_from_padding_mask(
            right_padded_prompt_tokens_batched != 0
        )
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

        causal_mask = rlhf.get_causal_mask_from_padding_mask(
            mixed_padded_prompt_tokens != 0
        )
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

        causal_mask = rlhf.get_causal_mask_from_padding_mask(
            mixed_padded_prompt_tokens_batched != 0
        )
        torch.testing.assert_close(causal_mask, expected_causal_mask, atol=0, rtol=0)


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from tests.test_utils import fixed_init_model

from torchtune import utils
from torchtune.models.llama2 import llama2


class TestTextGenerate:
    """
    Test class for text generation functionality.
    """

    @pytest.fixture
    def generation_model(self, dtype=torch.float32):
        model = llama2(
            vocab_size=4_000,
            embed_dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            max_seq_len=2048,
        )
        fixed_init_model(model)
        model.setup_caches(batch_size=1, dtype=dtype)
        model.eval()
        return model

    @pytest.fixture
    def generation_model_no_kv_cache(self, dtype=torch.float32):
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
    def generation_model_batched(self, dtype=torch.float32):
        model = llama2(
            vocab_size=4_000,
            embed_dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            max_seq_len=2048,
        )
        fixed_init_model(model)
        model.setup_caches(batch_size=2, dtype=dtype)
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
        Pytest fixture to create a list of prompt tokens for testing.
        """
        return torch.arange(2, 10).repeat(2, 1)

    def test_sample_consistency(self):
        """
        Test token sampling produces the right output.
        """
        # set all probabilities except for token_id=100 to 0
        logits = torch.zeros(2000)
        logits[100] = 1

        token = sample(logits, temperature=1, top_k=1)
        assert token.item() == 100

    @pytest.mark.parametrize(
        "model1,model2,prompt",
        [
            ("generation_model", "generation_model", "prompt_tokens"),
            ("generation_model", "generation_model_no_kv_cache", "prompt_tokens"),
            (
                "generation_model_batched",
                "generation_model_batched",
                "prompt_tokens_batched",
            ),
            (
                "generation_model_batched",
                "generation_model_no_kv_cache",
                "prompt_tokens_batched",
            ),
        ],
    )
    def test_reproducibility(self, request, model1, model2, prompt):
        """
        Test to check if the `generate` function produces the same output when run with the same
        inputs and a fixed seed. This should work regardless of batched input or kv cache.
        """

        model1 = request.getfixturevalue(model1)
        model2 = request.getfixturevalue(model2)
        prompt = request.getfixturevalue(prompt)

        temperature = 0.6
        top_k = 100

        torch.manual_seed(42)
        outputs_first = utils.generate(
            model=model1,
            prompt=prompt,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
        )

        torch.manual_seed(42)
        outputs_second = utils.generate(
            model=model2,
            prompt=prompt,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
        )

        assert outputs_first == outputs_second

    def test_batched_generate(self, generation_model_batched, prompt_tokens_batched):
        """Test batched generation works as expected."""
        temperature = 0.6
        top_k = 100

        torch.manual_seed(42)

        output = utils.generate(
            model=generation_model_batched,
            prompt=prompt_tokens_batched,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
        )

        # The numbers here are the first 10 tokens generated by the model
        # with constantly initialized weights, a tensor input with range 2 through 10,
        # and the manual seed set to 42. They do not correspond to "recognizable" tokens.
        expected_output = [
            [
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                3987,
                3991,
                3953,
                3957,
                3983,
                3964,
                3928,
                3932,
                3986,
                3982,
            ],
            [
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                3958,
                3979,
                3934,
                3945,
                3993,
                3904,
                3950,
                3988,
                3948,
                3999,
            ],
        ]

        assert output == expected_output

    def test_stop_tokens(self, generation_model, prompt_tokens):
        """
        Test to check if the `generate` function produces the right output when stop tokens are
        provided.
        """
        temperature = 0.6
        top_k = 100

        # This is the first token generated by the model
        # so it should stop immediately
        stop_tokens = [3987]

        torch.manual_seed(42)

        outputs = utils.generate(
            model=generation_model,
            prompt=prompt_tokens,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=stop_tokens,
        )

        expected_output = [[2, 3, 4, 5, 6, 7, 8, 9, 3987]]

        assert outputs == expected_output

    def test_stop_tokens_batched(self, generation_model_batched, prompt_tokens_batched):
        """
        Test to check if the `generate` function produces the right output when stop tokens are
        provided, but this time in batched format.
        """
        temperature = 0.6
        top_k = 100

        # This is the first token generated by the model
        # so it should stop immediately
        stop_tokens = [3987, 3958]

        torch.manual_seed(42)

        outputs = utils.generate(
            model=generation_model_batched,
            prompt=prompt_tokens_batched,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=stop_tokens,
        )

        expected_output = [
            [2, 3, 4, 5, 6, 7, 8, 9, 3987],
            [2, 3, 4, 5, 6, 7, 8, 9, 3958],
        ]

        assert outputs == expected_output

    def test_stop_tokens_batched_uneven_stopping(
        self, generation_model_batched, prompt_tokens_batched
    ):
        """
        Test to check if the `generate` function produces the right output when stop tokens are
        provided, but this time in batched format. This time, seq 0 should hit a stop token before seq 1.
        We expect the output to be the length of seq 1, but the first seq should be truncated.
        """
        temperature = 0.6
        top_k = 100

        # This is the first token generated by the model
        # so it should stop immediately
        stop_tokens = [3987, 3979]

        torch.manual_seed(42)

        outputs = utils.generate(
            model=generation_model_batched,
            prompt=prompt_tokens_batched,
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=stop_tokens,
        )

        expected_output = [
            [2, 3, 4, 5, 6, 7, 8, 9, 3987, 0],
            [2, 3, 4, 5, 6, 7, 8, 9, 3958, 3979],
        ]

        assert outputs == expected_output
