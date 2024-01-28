# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torchtune.models.lora_llama2 import lora_llama2
from torchtune.modules.peft.peft_utils import (
    _get_adapter_params,
    _get_base_model_params,
    _set_trainable_params,
    AdapterModule,
)

N_LAYERS = 3
IN_DIM = 5
OUT_DIM = 10


class DummyAdapterModule(nn.Module, AdapterModule):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.adapter = nn.Linear(in_dim, out_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    @classmethod
    def _adapter_params(cls):
        return ["adapter"]

    def forward(self, x):
        return self.adapter(x) + self.non_adapter(x)


class DummyAdapterParentModel(nn.Module, AdapterModule):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dummy_adapter_module = DummyAdapterModule(in_dim, out_dim)
        self.parent_adapter = nn.Linear(in_dim, out_dim)
        self.parent_base_model = nn.Linear(in_dim, out_dim)

    @classmethod
    def _adapter_params(cls):
        return ["parent_adapter"]

    def forward(self, x):
        return (
            self.dummy_adapter_module(x)
            + self.parent_adapter(x)
            + self.parent_base_model(x)
        )


class DummyAdapterWithParameter(nn.Module, AdapterModule):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.adapter = nn.Parameter(torch.randn(out_dim, in_dim))
        self.linear = nn.Linear(in_dim, out_dim)

    @classmethod
    def _adapter_params(cls):
        return ["adapter"]

    def forward(self, x):
        return (self.adapter @ x) + self.non_adapter(x)


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
                f"{i}.dummy_adapter_module.adapter.bias",
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
def dummy_param_model():
    return DummyAdapterWithParameter(IN_DIM, OUT_DIM)


@pytest.fixture
def dummy_param_model_expected_adapter_keys():
    return ["adapter"]


@pytest.fixture
def dummy_param_model_expected_base_model_keys():
    return ["linear.weight", "linear.bias"]


@pytest.fixture
def lora_llama2_model():
    return lora_llama2(
        lora_attn_modules=["q_proj", "v_proj"],
        vocab_size=50,
        num_layers=N_LAYERS,
        num_heads=4,
        num_kv_heads=2,
        embed_dim=64,
        max_seq_len=64,
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
    keys = ["tok_embeddings.weight", "output.weight", "norm.scale"]
    for i in range(N_LAYERS):
        keys.extend(
            [
                f"layers.{i}.sa_norm.scale",
                f"layers.{i}.attn.q_proj.linear.weight",
                f"layers.{i}.attn.k_proj.weight",
                f"layers.{i}.attn.v_proj.linear.weight",
                f"layers.{i}.attn.output_proj.weight",
                f"layers.{i}.mlp_norm.scale",
                f"layers.{i}.mlp.w1.weight",
                f"layers.{i}.mlp.w2.weight",
                f"layers.{i}.mlp.w3.weight",
            ]
        )
    return keys


class TestPeftUtils:
    @pytest.mark.parametrize(
        "model_name, expected_keys",
        [
            ("dummy_adapter_parent_model", "dummy_model_expected_adapter_keys"),
            ("dummy_param_model", "dummy_param_model_expected_adapter_keys"),
            ("lora_llama2_model", "lora_llama2_expected_adapter_keys"),
        ],
    )
    def test_get_adapter_params(self, request, model_name, expected_keys):
        model = request.getfixturevalue(model_name)
        adapter_params = _get_adapter_params(model)
        expected = request.getfixturevalue(expected_keys)
        assert set(expected) == set(adapter_params.keys())

    @pytest.mark.parametrize(
        "model_name, expected_keys",
        [
            ("dummy_adapter_parent_model", "dummy_model_expected_base_model_keys"),
            ("dummy_param_model", "dummy_param_model_expected_base_model_keys"),
            ("lora_llama2_model", "lora_llama2_expected_base_model_keys"),
        ],
    )
    def test_get_base_model_params(self, request, model_name, expected_keys):
        model = request.getfixturevalue(model_name)
        base_model_params = _get_base_model_params(model)
        expected = request.getfixturevalue(expected_keys)
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
                "dummy_param_model",
                "dummy_param_model_expected_adapter_keys",
                "dummy_param_model_expected_base_model_keys",
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
        expected_trainable = request.getfixturevalue(expected_trainable_keys)
        expected_frozen = request.getfixturevalue(expected_frozen_keys)
        _set_trainable_params(model)
        for k, v in model.named_parameters():
            if k in expected_trainable:
                assert v.requires_grad
            elif k in expected_frozen:
                assert not v.requires_grad
            else:
                raise AssertionError(f"{k} not in expected keys")
