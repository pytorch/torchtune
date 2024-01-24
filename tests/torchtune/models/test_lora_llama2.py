# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtune.models.lora_llama2 import lora_llama_self_attention
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
    def in_dim(self) -> int:
        return 64

    @pytest.fixture
    def inputs(self) -> torch.Tensor:
        inputs = torch.randn(BSZ, SEQ_LEN, EMBED_DIM)
        return inputs

    def get_lora_llama_self_attention(self, lora_modules):
        return lora_llama_self_attention(
            lora_modules=lora_modules,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            max_seq_len=MAX_SEQ_LEN,
            lora_rank=RANK,
            lora_alpha=ALPHA,
        )

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
        fixed_init_model(lora_llama_sa)
        actual = lora_llama_sa(inputs)
        assert_expected(actual.shape, (BSZ, SEQ_LEN, EMBED_DIM))
        assert_expected(actual.mean(), expected, atol=1e-4, rtol=1e-6)
