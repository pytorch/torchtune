# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Callable, List

import numpy as np

import pytest
import torch

from tests.test_utils import init_weights_with_constant

from torchtune import utils
from torchtune.models.llama2 import llama2
from torchtune.utils._generation import sample


@pytest.fixture(autouse=True)
def prevent_leaking_rng():
    # Prevent each test from leaking the rng to all other test when they call
    # torch.manual_seed() or random.seed() or np.random.seed().

    torch_rng_state = torch.get_rng_state()
    builtin_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()

    yield

    torch.set_rng_state(torch_rng_state)
    random.setstate(builtin_rng_state)
    np.random.set_state(numpy_rng_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)


@pytest.fixture(autouse=True)
def random_seed():
    utils.set_seed(42)


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
        init_weights_with_constant(model)
        model.setup_caches(max_batch_size=1, dtype=dtype)
        model.eval()
        return model

    @pytest.fixture
    def prompt_tokens(self) -> List[int]:
        """
        Pytest fixture to create a list of prompt tokens for testing.
        Returns:
            A list of prompt tokens.
        """
        return torch.arange(2, 10)

    def test_sample_consistency(self):
        """
        Test token sampling produces the right output.
        """
        # set all probabilities except for token_id=100 to 0
        logits = torch.zeros(2000)
        logits[100] = 1

        token = sample(logits, temperature=1, top_k=1)
        assert token.item() == 100

    def test_reproducibility(self, generation_model, prompt_tokens):
        """
        Test to check if the `generate` function produces the same output when run with the same
        inputs and a fixed seed.
        """
        temperature = 0.6
        top_k = 100

        torch.manual_seed(42)
        outputs_first = utils.generate(
            model=generation_model,
            prompt=prompt_tokens.unqueeze(0),
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
        )

        torch.manual_seed(42)
        outputs_second = utils.generate(
            model=generation_model,
            prompt=prompt_tokens.unqueeze(0),
            max_generated_tokens=10,
            temperature=temperature,
            top_k=top_k,
        )

        assert outputs_first == outputs_second
