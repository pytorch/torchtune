# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from itertools import chain

import pytest
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchtune.models.llama2 import llama2
from torchtune.models.lora_llama2 import lora_llama2
from torchtune.modules.peft.lora import LoRALinear, reset_lora_params
from torchtune.modules.peft.peft_utils import (
    _get_base_model_params,
    AdapterModule,
    get_adapter_params,
    lora_fsdp_init,
    lora_fsdp_wrap_policy,
    set_trainable_params,
    validate_state_dict_for_lora,
)
from torchtune.modules.transformer import TransformerDecoderLayer

from tests.test_utils import single_box_init

N_LAYERS = 3
IN_DIM = 5
OUT_DIM = 10
VOCAB_SIZE = 50
NUM_HEADS = 4
NUM_KV_HEADS = 2
EMBED_DIM = 64
MAX_SEQ_LEN = 64


def _get_n_lora_and_tformer_layers(model):
    num_lora_ab = 0
    num_transformer_layers = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            num_nested_linears = len(
                [m for m in module.modules() if isinstance(m, nn.Linear)]
            )
            num_lora_ab += num_nested_linears
        if isinstance(module, TransformerDecoderLayer):
            num_transformer_layers += 1

    return num_lora_ab, num_transformer_layers


class DummyAdapterModule(nn.Module, AdapterModule):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.adapter = nn.Linear(in_dim, out_dim, bias=False)
        self.linear = nn.Linear(in_dim, out_dim)

    def adapter_params(self):
        return ["adapter.weight"]

    def forward(self, x):
        return self.adapter(x) + self.non_adapter(x)


class DummyAdapterParentModel(nn.Module, AdapterModule):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dummy_adapter_module = DummyAdapterModule(in_dim, out_dim)
        self.parent_adapter = nn.Linear(in_dim, out_dim)
        self.parent_base_model = nn.Linear(in_dim, out_dim)

    def adapter_params(self):
        return ["parent_adapter.weight", "parent_adapter.bias"]

    def forward(self, x):
        return (
            self.dummy_adapter_module(x)
            + self.parent_adapter(x)
            + self.parent_base_model(x)
        )


@pytest.fixture
def dummy_adapter_parent_model():
    return nn.ModuleList(
        [DummyAdapterParentModel(IN_DIM, OUT_DIM) for _ in range(N_LAYERS)]
    )


@pytest.fixture
def dummy_model_expected_adapter_keys():
    keys = []
    for i in range(N_LAYERS):
        keys.extend(
            [
                f"{i}.parent_adapter.weight",
                f"{i}.parent_adapter.bias",
                f"{i}.dummy_adapter_module.adapter.weight",
            ]
        )
    return keys


@pytest.fixture
def dummy_model_expected_base_model_keys():
    keys = []
    for i in range(N_LAYERS):
        keys.extend(
            [
                f"{i}.parent_base_model.weight",
                f"{i}.parent_base_model.bias",
                f"{i}.dummy_adapter_module.linear.weight",
                f"{i}.dummy_adapter_module.linear.bias",
            ]
        )
    return keys


@pytest.fixture
def lora_llama2_model():
    return lora_llama2(
        lora_attn_modules=["q_proj", "v_proj"],
        vocab_size=VOCAB_SIZE,
        num_layers=N_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        embed_dim=EMBED_DIM,
        max_seq_len=MAX_SEQ_LEN,
        lora_rank=4,
        lora_alpha=1.0,
    )


@pytest.fixture
def lora_llama2_expected_adapter_keys():
    keys = []
    for i in range(N_LAYERS):
        keys.extend(
            [
                f"layers.{i}.attn.q_proj.lora_a.weight",
                f"layers.{i}.attn.q_proj.lora_b.weight",
                f"layers.{i}.attn.v_proj.lora_a.weight",
                f"layers.{i}.attn.v_proj.lora_b.weight",
            ]
        )
    return keys


@pytest.fixture
def lora_llama2_expected_base_model_keys():

    base_model = llama2(
        vocab_size=VOCAB_SIZE,
        num_layers=N_LAYERS,
        num_heads=NUM_KV_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        embed_dim=EMBED_DIM,
        max_seq_len=MAX_SEQ_LEN,
    )
    return base_model.state_dict().keys()


class TestPeftUtils:
    @pytest.mark.parametrize(
        "model_name, expected_keys",
        [
            ("dummy_adapter_parent_model", "dummy_model_expected_adapter_keys"),
            ("lora_llama2_model", "lora_llama2_expected_adapter_keys"),
        ],
    )
    def test_get_adapter_params(self, request, model_name, expected_keys):
        model = request.getfixturevalue(model_name)
        adapter_params = get_adapter_params(model)
        expected = request.getfixturevalue(expected_keys)
        assert set(expected) == set(adapter_params.keys())

    @pytest.mark.parametrize(
        "model_name, expected_keys",
        [
            ("dummy_adapter_parent_model", "dummy_model_expected_base_model_keys"),
            ("lora_llama2_model", "lora_llama2_expected_base_model_keys"),
        ],
    )
    def test_get_base_model_params(self, request, model_name, expected_keys):
        model = request.getfixturevalue(model_name)
        base_model_params = _get_base_model_params(model)
        expected = request.getfixturevalue(expected_keys)
        print(set(expected).difference(set(base_model_params.keys())))
        print(set(base_model_params.keys()).difference(set(expected)))
        assert set(expected) == set(base_model_params.keys())

    @pytest.mark.parametrize(
        "model_name, expected_trainable_keys, expected_frozen_keys",
        [
            (
                "dummy_adapter_parent_model",
                "dummy_model_expected_adapter_keys",
                "dummy_model_expected_base_model_keys",
            ),
            (
                "lora_llama2_model",
                "lora_llama2_expected_adapter_keys",
                "lora_llama2_expected_base_model_keys",
            ),
        ],
    )
    def test_set_trainable_params(
        self, request, model_name, expected_trainable_keys, expected_frozen_keys
    ):
        model = request.getfixturevalue(model_name)
        adapter_params = get_adapter_params(model)
        expected_trainable = request.getfixturevalue(expected_trainable_keys)
        expected_frozen = request.getfixturevalue(expected_frozen_keys)
        set_trainable_params(model, adapter_params)
        for k, v in model.named_parameters():
            if k in expected_trainable:
                assert v.requires_grad
            elif k in expected_frozen:
                assert not v.requires_grad
            else:
                raise AssertionError(f"{k} not in expected keys")

    @pytest.mark.parametrize(
        "missing_keys, unexpected_keys, lora_modules, expected_result",
        [
            (["a", "q_proj.weight"], [], ["q_proj"], "Missing"),
            ([], ["b"], ["q_proj", "k_proj"], "Unexpected"),
            (
                ["q_proj.weight", "output_proj.weight"],
                [],
                ["q_proj", "output_proj"],
                "",
            ),
        ],
    )
    def test_validate_state_dict_for_lora(
        self, missing_keys, unexpected_keys, lora_modules, expected_result
    ):
        if expected_result:
            with pytest.raises(AssertionError, match=expected_result):
                validate_state_dict_for_lora(
                    missing_keys, unexpected_keys, lora_modules
                )
        else:
            validate_state_dict_for_lora(missing_keys, unexpected_keys, lora_modules)

    def test_lora_fsdp_wrap(self):
        with torch.device("meta"):
            model = lora_llama2(
                lora_attn_modules=["q_proj", "v_proj"],
                vocab_size=VOCAB_SIZE,
                num_layers=N_LAYERS,
                num_heads=NUM_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                embed_dim=EMBED_DIM,
                max_seq_len=MAX_SEQ_LEN,
                lora_rank=4,
                lora_alpha=1.0,
            )

        reset_lora_params(model, device=torch.device("cpu"))
        adapter_params = get_adapter_params(model)
        set_trainable_params(model, adapter_params)
        num_lora_ab, num_transformer_layers = _get_n_lora_and_tformer_layers(model)
        with single_box_init():
            lora_wrap_policy = lora_fsdp_wrap_policy(
                modules_to_wrap={TransformerDecoderLayer}
            )

            lora_init = partial(lora_fsdp_init, device=torch.device("cpu"))
            wrapped_lora = FSDP(
                model,
                auto_wrap_policy=lora_wrap_policy,
                device_id=torch.device("cpu"),
                param_init_fn=lora_init,
            )

            # After FSDP wrap, nothing should be left on meta device, and LoRA params
            # should be initialized.
            for p in chain(wrapped_lora.parameters(), wrapped_lora.buffers()):
                assert not p.is_meta

            for m in wrapped_lora.modules():
                if isinstance(m, LoRALinear):
                    assert m._lora_params_initialized

            # Total # FSDP modules should be num_transformer + num_lora_ab + 1
            total_fsdp_submodules = len([m for m in FSDP.fsdp_modules(wrapped_lora)])
            assert total_fsdp_submodules == (num_lora_ab + num_transformer_layers + 1)
            # LoRA a & b linears should be individually wrapped.
            # And TransformerDecoderLayers should be individually wrapped.
            for fsdp_submodule in FSDP.fsdp_modules(wrapped_lora):
                if isinstance(fsdp_submodule.module, nn.Linear):
                    num_lora_ab -= 1
                elif isinstance(fsdp_submodule.module, TransformerDecoderLayer):
                    num_transformer_layers -= 1
            assert num_lora_ab == 0
            assert num_transformer_layers == 0

    def test_lora_meta_device_init_fsdp(self):
        with torch.device("meta"):
            lora = lora_llama2(
                lora_attn_modules=["q_proj", "v_proj"],
                vocab_size=VOCAB_SIZE,
                num_layers=N_LAYERS,
                num_heads=NUM_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                embed_dim=EMBED_DIM,
                max_seq_len=MAX_SEQ_LEN,
                lora_rank=4,
                lora_alpha=1.0,
            )

        for m in lora.modules():
            lora_fsdp_init(m, device=torch.device("cpu"))
        # No params should be left on meta device
        for n, p in lora.named_parameters():
            assert not p.is_meta, f"parameter {n} is still on meta device!"
        # Neither should buffers
        for n, b in lora.named_buffers():
            assert not b.is_meta, f"buffer {n} is still on meta device!"
