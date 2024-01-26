# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torchtune.models.lora_llama2 import _lora_llama_self_attention, lora_llama2
from torchtune.utils.seed import set_seed

from tests.test_utils import assert_expected, fixed_init_model

RANK = 4
ALPHA = 1.0
BSZ = 2
SEQ_LEN = 32
EMBED_DIM = 64
NUM_HEADS = 4
NUM_KV_HEADS = 2
MAX_SEQ_LEN = 64


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestLoRALlamaSelfAttention:
    @pytest.fixture
    def inputs(self) -> torch.Tensor:
        inputs = torch.randn(BSZ, SEQ_LEN, EMBED_DIM)
        return inputs

    def get_lora_llama_self_attention(self, lora_modules):
        lora_llama_sa = _lora_llama_self_attention(
            lora_modules=lora_modules,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            max_seq_len=MAX_SEQ_LEN,
            lora_rank=RANK,
            lora_alpha=ALPHA,
        )
        fixed_init_model(lora_llama_sa)
        return lora_llama_sa

    def test_empty_lora_modules(self):
        with pytest.raises(ValueError, match="Must pass one or more of"):
            _ = self.get_lora_llama_self_attention([])

    @pytest.mark.parametrize(
        "lora_modules, expected",
        [
            (["q_proj", "v_proj"], torch.tensor(51.3152)),
            (["q_proj", "k_proj", "v_proj", "output_proj"], torch.tensor(79.8887)),
            (["k_proj"], torch.tensor(45.9261)),
        ],
    )
    def test_forward(self, inputs, lora_modules, expected):
        lora_llama_sa = self.get_lora_llama_self_attention(lora_modules)
        actual = lora_llama_sa(inputs)
        assert_expected(actual.shape, (BSZ, SEQ_LEN, EMBED_DIM))
        assert_expected(actual.mean(), expected, atol=1e-4, rtol=1e-6)


class TestLoRALlama2:
    @pytest.fixture
    def vocab_size(self):
        return 50

    @pytest.fixture
    def inputs(self, vocab_size):
        return torch.randint(low=0, high=vocab_size, size=(BSZ, SEQ_LEN))

    def get_lora_llama2(self, lora_modules, vocab_size):
        num_layers = 3
        model = lora_llama2(
            lora_attn_modules=lora_modules,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            embed_dim=EMBED_DIM,
            max_seq_len=MAX_SEQ_LEN,
            lora_rank=RANK,
            lora_alpha=ALPHA,
        )
        # To make final outputs less trivial
        model.norm = nn.Identity()
        fixed_init_model(model)
        return model

    @pytest.mark.parametrize(
        "lora_modules, expected",
        [
            (["q_proj", "v_proj"], torch.tensor(5638870.5)),
            (["q_proj", "k_proj", "v_proj", "output_proj"], torch.tensor(5684272.5)),
            (["k_proj"], torch.tensor(5608697.5)),
        ],
    )
    def test_forward(self, vocab_size, inputs, lora_modules, expected):
        model = self.get_lora_llama2(lora_modules, vocab_size)
        fixed_init_model(model)
        actual = model(inputs)
        assert_expected(actual.shape, (BSZ, SEQ_LEN, vocab_size))
        assert_expected(actual.mean(), expected, atol=1e-4, rtol=1e-6)
