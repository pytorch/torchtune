# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List

import pytest

import torch
from torchtune.modules import (
    CausalSelfAttention,
    FeedForward,
    RotaryPositionalEmbeddings,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from torchtune.models.llama2 import Llama2
from torchtune.utils.env import seed
from torchtune.utils.generation import GenerationUtils

from tests.test_utils import assert_expected, init_weights_with_constant


@pytest.fixture(autouse=True)
def set_seed():
    seed(42)


_test_pad_id = -1
_test_eos_id = 2


def _make_generate(decoder_lm) -> Callable:
    return GenerationUtils(
        decoder_lm=decoder_lm, eos_id=_test_eos_id, pad_id=_test_pad_id
    ).generate


def _token_lists_to_tensor(self, token_lists: List[List[int]]) -> torch.LongTensor:
    max_seq_len = max(len(tok) for tok in token_lists)
    token_lists[:] = [
        tok + [_test_pad_id] * (max_seq_len - len(tok)) for tok in token_lists
    ]
    return torch.tensor(token_lists, dtype=torch.long)


class TestTextGenerate:
    """
    Test class for text generation functionality.
    """

    @property
    def _batch_size(self):
        return 2

    @pytest.fixture
    def generation_model(self):
        """
        A dummy model to test `generate` API
        """
        model = Llama2(
            vocab_size=4_000,
            embed_dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=None,
            max_seq_len=2048,
        )
        init_weights_with_constant(model)
        model.eval()
        return model

    @pytest.fixture
    def generation_model_kv_cache(self):
        """
        A dummy model to test incremental decoding portion of `generate` API
        w/kv-caching enabled.
        """
        generation_model_kv_cache = Llama2(
            vocab_size=4_000,
            embed_dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=None,
            max_seq_len=2048,
            max_batch_size=2,
        )
        init_weights_with_constant(generation_model_kv_cache)
        generation_model_kv_cache.eval()
        return generation_model_kv_cache

    @pytest.fixture
    def prompt_tokens(self) -> List[int]:
        """
        Pytest fixture to create a list of prompt tokens for testing.
        Returns:
            A list of prompt tokens.
        """
        return [list(range(2, 10)) for _ in range(self._batch_size)]

    def test_different_len_prompts_in_batch(self, generation_model):
        """
        Test to check if the `generate` function can handle prompts of different lengths in a batch.
        """
        prompt_tokens = [
            [1],
            [8, 9],
            [4, 5, 6],
            [7, 8, 9, 20],
        ]
        min_gen_len = 1
        max_gen_len = 1
        temperature = 1.0
        top_p = 1.0
        top_k = 0
        generate = _make_generate(generation_model)
        outputs_actual, _ = generate(
            prompt_tokens=prompt_tokens,
            min_gen_len=min_gen_len,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            incremental_decode=False,
        )
        # Since keep_prompt=True by default, each generation should have
        # its prompt at the beginning.
        expected_prompt_lens = [len(prompt) for prompt in prompt_tokens]
        assert len(expected_prompt_lens) == len(outputs_actual)
        for i, (expected_len, generation) in enumerate(
            zip(expected_prompt_lens, outputs_actual)
        ):
            generation_tokens = generation.tolist()
            expected_prompt = generation_tokens[:expected_len]
            assert_expected(expected_prompt, prompt_tokens[i])
            for tok in generation_tokens:
                assert tok not in (_test_pad_id,)

    def test_no_keep_prompt(self, generation_model):
        """
        Test to check if the `generate` function works correctly when `keep_prompt` is set to False.
        """
        prompt_tokens = [
            [1],
            [8, 9],
            [4, 5, 6],
            [7, 8, 9, 20],
        ]
        min_gen_len = 1
        max_gen_len = 1
        temperature = 1.0
        top_p = 1.0
        top_k = 0
        generate = _make_generate(generation_model)
        outputs_actual, _ = generate(
            prompt_tokens=prompt_tokens,
            min_gen_len=min_gen_len,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            keep_prompt=False,
            incremental_decode=False,
        )
        for generation in outputs_actual:
            generation = generation.tolist()
            assert_expected(len(generation), max_gen_len)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
    def test_cuda_device(self, generation_model, prompt_tokens):
        """
        Test to check if the `generate` function reutnrs outputs on the expected CUDA device.
        """
        min_gen_len = 1
        max_gen_len = 1
        generation_model.cuda()
        generate = _make_generate(generation_model)
        outputs = generate(
            prompt_tokens=prompt_tokens,
            min_gen_len=min_gen_len,
            max_gen_len=max_gen_len,
            logprobs=True,
            device="cuda",
            incremental_decode=False,
        )
        assert outputs[0].device == torch.device("cuda", 0)
        assert outputs[1].device == torch.device("cuda", 0)

    def test_token_logprobs(self, generation_model, prompt_tokens):
        """
        Test to check if the `generate` function returns expected type for token_logprobs.
        """
        min_gen_len = 1
        max_gen_len = 1
        generate = _make_generate(generation_model)
        outputs = generate(
            prompt_tokens=prompt_tokens,
            min_gen_len=min_gen_len,
            max_gen_len=max_gen_len,
            logprobs=True,
            incremental_decode=False,
        )
        assert_expected(outputs[0].shape, outputs[1].shape)
        assert isinstance(outputs[1], torch.FloatTensor)

    def test_kv_cache_incremental_decode_parity(
        self, generation_model, generation_model_kv_cache, prompt_tokens
    ):
        """
        Test to check if the `generate` function produces the same output when run with and without
        incremental decoding, where we use a kv-caching model with incremental_decode=True.
        """
        min_gen_len = 1
        max_gen_len = 20
        temperature = 1.0
        top_p = 1.0
        top_k = 0
        generate = _make_generate(generation_model)
        generate_kv_cache = _make_generate(generation_model_kv_cache)
        outputs, _ = generate(
            prompt_tokens=prompt_tokens,
            min_gen_len=min_gen_len,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            keep_prompt=False,
            incremental_decode=False,
        )
        outputs_kv_cache, _ = generate_kv_cache(
            prompt_tokens=prompt_tokens,
            min_gen_len=min_gen_len,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            keep_prompt=False,
            incremental_decode=True,
        )
        assert outputs.tolist() == outputs_kv_cache.tolist()
