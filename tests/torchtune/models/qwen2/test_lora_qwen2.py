# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from tests.test_utils import assert_expected, fixed_init_model
from torch import nn
from torchtune.models.qwen2 import lora_qwen2, qwen2
from torchtune.models.qwen2._component_builders import lora_qwen2_self_attention
from torchtune.training.seed import set_seed

RANK = 4
ALPHA = 1.0
BSZ = 2
SEQ_LEN = 32
EMBED_DIM = 64
INTERMEDIATE_DIM = 168
NUM_HEADS = 4
NUM_KV_HEADS = 2
MAX_SEQ_LEN = 64


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestLoRAQwen2SelfAttention:
    @pytest.fixture
    def inputs(self) -> torch.Tensor:
        inputs = torch.randn(BSZ, SEQ_LEN, EMBED_DIM)
        return inputs

    def get_lora_qwen2_self_attention(self, lora_modules):
        lora_qwen2 = lora_qwen2_self_attention(
            lora_modules=lora_modules,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            max_seq_len=MAX_SEQ_LEN,
            lora_rank=RANK,
            lora_alpha=ALPHA,
        )
        fixed_init_model(lora_qwen2)
        return lora_qwen2

    def test_empty_lora_modules(self):
        with pytest.raises(ValueError, match="Must pass one or more of"):
            _ = self.get_lora_qwen2_self_attention([])

    @pytest.mark.parametrize(
        "lora_modules, expected",
        [
            (["q_proj", "v_proj"], torch.tensor(83.6596)),
            (["q_proj", "k_proj", "v_proj", "output_proj"], torch.tensor(129.4454)),
            (["k_proj"], torch.tensor(69.3473)),
        ],
    )
    def test_forward(self, inputs, lora_modules, expected):
        lora_qwen2_sa = self.get_lora_qwen2_self_attention(lora_modules)
        actual = lora_qwen2_sa(inputs, inputs)
        assert_expected(actual.shape, (BSZ, SEQ_LEN, EMBED_DIM))
        assert_expected(actual.mean(), expected, atol=1e-4, rtol=1e-6)


class TestLoRAQwen2:
    @pytest.fixture
    def vocab_size(self):
        return 50

    @pytest.fixture
    def inputs(self, vocab_size):
        return torch.randint(low=0, high=vocab_size, size=(BSZ, SEQ_LEN))

    def get_lora_qwen2(
        self,
        lora_modules,
        apply_lora_to_mlp,
        apply_lora_to_output,
        vocab_size,
        reset_norm=True,
        quantize_base=False,
        embed_dim=EMBED_DIM,
        dtype=None,
    ):
        num_layers = 3
        model = lora_qwen2(
            lora_attn_modules=lora_modules,
            apply_lora_to_mlp=apply_lora_to_mlp,
            apply_lora_to_output=apply_lora_to_output,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            embed_dim=embed_dim,
            intermediate_dim=INTERMEDIATE_DIM,
            max_seq_len=MAX_SEQ_LEN,
            lora_rank=RANK,
            lora_alpha=ALPHA,
            quantize_base=quantize_base,
        )
        # To make final outputs less trivial
        if reset_norm:
            model.norm = nn.Identity()

        # dtype=None means to just read dtype from parameters
        # in the model. This dtype is set explicitly to bf16 currently
        # when initializing QLoRA models, as ops such as `arange` aren't
        # yet supported with the actual nf4 tensor dtype yet.
        fixed_init_model(model, dtype=dtype)

        return model

    def get_ref_qwen2(self, vocab_size, embed_dim=EMBED_DIM):
        num_layers = 3
        model = qwen2(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            embed_dim=embed_dim,
            intermediate_dim=INTERMEDIATE_DIM,
            max_seq_len=MAX_SEQ_LEN,
        )
        return model

    @pytest.mark.parametrize(
        "lora_modules, apply_lora_to_mlp, apply_lora_to_output, expected",
        [
            (["q_proj", "v_proj"], False, False, torch.tensor(3736558.0)),
            (
                ["q_proj", "k_proj", "v_proj", "output_proj"],
                True,
                False,
                torch.tensor(13962364.0),
            ),
            (["k_proj"], True, True, torch.tensor(21335964.0)),
        ],
    )
    def test_forward(
        self,
        vocab_size,
        inputs,
        lora_modules,
        apply_lora_to_mlp,
        apply_lora_to_output,
        expected,
    ):
        model = self.get_lora_qwen2(
            lora_modules, apply_lora_to_mlp, apply_lora_to_output, vocab_size
        )
        actual = model(inputs)
        assert_expected(actual.shape, (BSZ, SEQ_LEN, vocab_size))
        assert_expected(actual.mean(), expected, atol=1e-4, rtol=1e-6)

    @pytest.mark.parametrize(
        "lora_modules, apply_lora_to_mlp, apply_lora_to_output",
        [
            (["q_proj", "v_proj"], True, False),
            (["q_proj", "k_proj", "v_proj", "output_proj"], False, False),
            (["k_proj"], True, True),
        ],
    )
    def test_lora_qwen2_state_dict_parity(
        self, lora_modules, apply_lora_to_mlp, apply_lora_to_output, vocab_size
    ):
        lora_qwen2 = self.get_lora_qwen2(
            lora_modules,
            apply_lora_to_mlp,
            apply_lora_to_output,
            vocab_size,
            reset_norm=False,
        )
        ref_qwen2 = self.get_ref_qwen2(vocab_size)
        # Ensure ref_qwen2 state_dict can be loaded into lora_qwen2 with only "lora"
        # keys missing.
        ref_qwen2_state_dict = ref_qwen2.state_dict()
        missing, unexpected = lora_qwen2.load_state_dict(
            ref_qwen2_state_dict, strict=False
        )
        assert not unexpected
        assert all(["lora" in key for key in missing])
