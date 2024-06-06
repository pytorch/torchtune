# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest
import torch

from tests.test_utils import assert_expected, fixed_init_model
from torch import nn
from torchao.dtypes.nf4tensor import NF4Tensor
from torchtune import utils
from torchtune.models.phi3 import lora_phi3, phi3
from torchtune.models.phi3._component_builders import lora_phi3_self_attention
from torchtune.modules.peft import LoRALinear
from torchtune.modules.peft.peft_utils import get_merged_lora_ckpt
from torchtune.utils.seed import set_seed

RANK = 4
ALPHA = 1.0
BSZ = 2
SEQ_LEN = 32
EMBED_DIM = 64
INTER_DIM = 128
NUM_HEADS = 4
NUM_KV_HEADS = 2
MAX_SEQ_LEN = 64


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestLoRAPhi3SelfAttention:
    @pytest.fixture
    def inputs(self) -> torch.Tensor:
        inputs = torch.randn(BSZ, SEQ_LEN, EMBED_DIM)
        return inputs

    def get_lora_phi_self_attention(self, lora_modules):
        lora_phi_sa = lora_phi3_self_attention(
            lora_modules=lora_modules,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            max_seq_len=MAX_SEQ_LEN,
            lora_rank=RANK,
            lora_alpha=ALPHA,
        )
        fixed_init_model(lora_phi_sa)
        return lora_phi_sa

    def test_empty_lora_modules(self):
        with pytest.raises(ValueError, match="Must pass one or more of"):
            _ = self.get_lora_phi_self_attention([])

    @pytest.mark.parametrize(
        "lora_modules, expected",
        [
            (["q_proj", "v_proj"], torch.tensor(51.88655)),
            (["q_proj", "k_proj", "v_proj", "output_proj"], torch.tensor(75.80934)),
            (["k_proj"], torch.tensor(44.00425)),
        ],
    )
    def test_forward(self, inputs, lora_modules, expected):
        lora_phi_sa = self.get_lora_phi_self_attention(lora_modules)
        actual = lora_phi_sa(inputs)
        assert_expected(actual.shape, (BSZ, SEQ_LEN, EMBED_DIM))
        assert_expected(actual.mean(), expected, atol=1e-4, rtol=1e-6)


class TestLoRAPhi3:
    @pytest.fixture
    def vocab_size(self):
        return 50

    @pytest.fixture
    def inputs(self, vocab_size):
        return torch.randint(low=0, high=vocab_size, size=(BSZ, SEQ_LEN))

    def get_lora_phi3(
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
        model = lora_phi3(
            lora_attn_modules=lora_modules,
            apply_lora_to_mlp=apply_lora_to_mlp,
            apply_lora_to_output=apply_lora_to_output,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            embed_dim=embed_dim,
            intermediate_dim=INTER_DIM,
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

    def get_ref_phi3(self, vocab_size, embed_dim=EMBED_DIM):
        num_layers = 3
        model = phi3(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            embed_dim=embed_dim,
            intermediate_dim=INTER_DIM,
            max_seq_len=MAX_SEQ_LEN,
        )
        return model

    @pytest.mark.parametrize(
        "lora_modules, apply_lora_to_mlp, apply_lora_to_output, expected",
        [
            (["q_proj", "v_proj"], False, False, torch.tensor(2869687.75)),
            (
                ["q_proj", "k_proj", "v_proj", "output_proj"],
                True,
                False,
                torch.tensor(10674772.0),
            ),
            (["k_proj"], True, True, torch.tensor(16285834.0)),
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
        model = self.get_lora_phi3(
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
    def test_lora_phi3_state_dict_parity(
        self, lora_modules, apply_lora_to_mlp, apply_lora_to_output, vocab_size
    ):
        lora_phi = self.get_lora_phi3(
            lora_modules,
            apply_lora_to_mlp,
            apply_lora_to_output,
            vocab_size,
            reset_norm=False,
        )
        ref_phi = self.get_ref_phi3(vocab_size)
        # Ensure ref_phi state_dict can be loaded into lora_phi with only "lora"
        # keys missing.
        ref_phi_state_dict = ref_phi.state_dict()
        missing, unexpected = lora_phi.load_state_dict(ref_phi_state_dict, strict=False)
        assert not unexpected
        assert all(["lora" in key for key in missing])

    def test_lora_linear_quantize_base(self):
        model = self.get_lora_phi3(
            lora_modules=["q_proj", "v_proj", "k_proj", "output_proj"],
            apply_lora_to_mlp=True,
            # quantize_base
            apply_lora_to_output=False,
            vocab_size=50,
            quantize_base=True,
            embed_dim=512,
            dtype=torch.bfloat16,
        )
        for module in model.modules():
            if isinstance(module, LoRALinear):
                assert module._quantize_base

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_qlora_phi3_parity(self, dtype, inputs):
        with utils.set_default_dtype(dtype):
            model_ref = self.get_lora_phi3(
                lora_modules=["q_proj", "v_proj", "k_proj", "output_proj"],
                apply_lora_to_mlp=True,
                apply_lora_to_output=False,
                vocab_size=50,
                quantize_base=False,
                embed_dim=512,
                dtype=dtype,
            )
            qlora = self.get_lora_phi3(
                lora_modules=["q_proj", "v_proj", "k_proj", "output_proj"],
                apply_lora_to_mlp=True,
                apply_lora_to_output=False,
                vocab_size=50,
                quantize_base=True,
                embed_dim=512,
                dtype=dtype,
            )
        qlora_sd = qlora.state_dict()
        model_ref.load_state_dict(qlora_sd)
        # Forward pass of model_ref and qlora should be the same, as QLoRA linear layers should use
        # a special linear operator that runs the compute in bf16, but only saves the 4 bit tensors
        # for backward.
        ref_output = model_ref(inputs)
        output = qlora(inputs)
        torch.testing.assert_close(ref_output, output)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_qlora_phi3_state_dict(self, dtype):
        with utils.set_default_dtype(dtype):
            model_ref = self.get_lora_phi3(
                lora_modules=["q_proj", "v_proj", "k_proj", "output_proj"],
                apply_lora_to_mlp=True,
                apply_lora_to_output=False,
                vocab_size=50,
                quantize_base=False,
                embed_dim=512,
                dtype=dtype,
            )
            high_prec_sd = model_ref.state_dict()
            for v in high_prec_sd.values():
                assert v.dtype == dtype

            # ensure quantized LoRA can load a bf16 state_dict
            qlora = self.get_lora_phi3(
                lora_modules=["q_proj", "v_proj", "k_proj", "output_proj"],
                apply_lora_to_mlp=True,
                apply_lora_to_output=False,
                vocab_size=50,
                quantize_base=True,
                embed_dim=512,
                dtype=dtype,
            )
            qlora.load_state_dict(high_prec_sd)
            # LoRALinear base weights should be nf4 still
            for module in qlora.modules():
                if isinstance(module, LoRALinear):
                    assert isinstance(module.weight, NF4Tensor)
            # saved state_dict should have bf16 weights.
            qlora_sd = qlora.state_dict()
            for v in qlora_sd.values():
                assert v.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_qlora_phi3_merged_state_dict(self, dtype):
        with utils.set_default_dtype(dtype):
            qlora = self.get_lora_phi3(
                lora_modules=["q_proj", "v_proj", "k_proj", "output_proj"],
                apply_lora_to_mlp=True,
                apply_lora_to_output=False,
                vocab_size=50,
                quantize_base=True,
                embed_dim=512,
                dtype=dtype,
                reset_norm=False,  # to ensure norm.scale key exists
            )

        qlora_sd = qlora.state_dict()
        # Ensure checkpoint merging produces bf16 tensors
        merged_ckpt = get_merged_lora_ckpt(deepcopy(qlora_sd), rank=RANK, alpha=ALPHA)
        for k, v in merged_ckpt.items():
            # paranoid check for both, as NF4Tensor had issue where NF4Tensor.dtype would return bf16
            assert not isinstance(v, NF4Tensor)
            if k == "":
                assert v.dtype == torch.float32
            else:
                assert v.dtype == dtype

        # Ensure checkpoint can be loaded into non-LoRA model
        with utils.set_default_dtype(dtype):
            phi3 = self.get_ref_phi3(vocab_size=50, embed_dim=512)

        phi3.load_state_dict(merged_ckpt)
