# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest
import torch
from torch import nn
from torchtune.models.llama2 import llama2, lora_llama2
from torchtune.modules.peft import LoRALinear
from torchtune.modules.peft.peft_utils import (
    _get_base_model_params,
    AdapterModule,
    get_adapter_params,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_state_dict_for_lora,
)

N_LAYERS = 3
IN_DIM = 5
OUT_DIM = 10
VOCAB_SIZE = 50
NUM_HEADS = 4
NUM_KV_HEADS = 2
EMBED_DIM = 64
MAX_SEQ_LEN = 64
RANK = 2
ALPHA = 1


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
def lora_llama2_model_all_keys(lora_llama2_model):
    return lora_llama2_model.state_dict().keys()


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
        (
            """
            lora_attn_modules,
            apply_lora_to_mlp,
            apply_lora_to_output,
            full_model_state_dict_keys,
            lora_state_dict_keys,
            base_model_state_dict_keys,
            expected
            """
        ),
        [
            (
                ["q_proj", "k_proj"],
                False,
                False,
                ["q_proj.lora_a.weight", "dummy_param.weight"],
                ["q_proj.lora_a.weight"],
                ["dummy_param.weight"],
                "",
            ),
            (
                ["v_proj"],
                False,
                False,
                ["param_a", "param_b"],
                None,
                ["param_a", "param_b"],
                "",
            ),
            (
                ["output_proj"],
                False,
                True,
                ["output_proj.weight", "output_proj.lora_a.weight"],
                ["output_proj.lora_a.weight"],
                ["output_proj.weight"],
                "",
            ),
            (["q_proj"], False, False, ["param_a"], [], [], "Missing non-LoRA"),
            (
                ["k_proj", "output_proj"],
                False,
                True,
                ["k_proj.lora_a.weight", "param_a"],
                ["k_proj.lora_a.weight", "param_a"],
                ["param_a"],
                "found in LoRA",
            ),
            (
                ["k_proj"],
                False,
                False,
                ["k_proj.lora_a.weight"],
                [],
                ["k_proj.lora_a.weight"],
                "found in base model",
            ),
            (
                ["k_proj"],
                False,
                False,
                ["k_proj.lora_a.weight"],
                [],
                None,
                "Missing LoRA",
            ),
            (["q_proj"], False, False, [], ["a"], ["a"], "overlapping"),
            (
                ["v_proj"],
                False,
                False,
                ["dummy_param.weight"],
                ["v_proj.lora_a.weight"],
                ["dummy_param.weight"],
                "Extra",
            ),
            (
                ["w1", "w2", "w3"],
                True,
                False,
                ["w1.lora_a.weight", "w2.weight", "q_proj.weight"],
                ["w1.lora_a.weight"],
                ["q_proj.weight"],
                "Missing non-LoRA key",
            ),
            (
                ["q_proj", "output"],
                False,
                True,
                [
                    "q_proj.lora_a",
                    "output.weight",
                    "output.lora_a",
                    "output_proj.lora_b",
                ],
                ["q_proj.lora_a", "output.lora_a", "output_proj.lora_b"],
                ["output.weight"],
                "Missing non-LoRA key",
            ),
            (
                ["q_proj", "v_proj"],
                False,
                False,
                "lora_llama2_model_all_keys",
                "lora_llama2_expected_adapter_keys",
                "lora_llama2_expected_base_model_keys",
                "",
            ),
        ],
    )
    def test_validate_lora_state_dict(
        self,
        request,
        lora_attn_modules,
        apply_lora_to_mlp,
        apply_lora_to_output,
        full_model_state_dict_keys,
        lora_state_dict_keys,
        base_model_state_dict_keys,
        expected,
    ):
        if isinstance(full_model_state_dict_keys, str):
            full_model_state_dict_keys = request.getfixturevalue(
                full_model_state_dict_keys
            )
        if isinstance(lora_state_dict_keys, str):
            lora_state_dict_keys = request.getfixturevalue(lora_state_dict_keys)
        if isinstance(base_model_state_dict_keys, str):
            base_model_state_dict_keys = request.getfixturevalue(
                base_model_state_dict_keys
            )
        if expected:
            with pytest.raises(AssertionError, match=expected):
                validate_state_dict_for_lora(
                    lora_attn_modules,
                    apply_lora_to_mlp,
                    apply_lora_to_output,
                    full_model_state_dict_keys=full_model_state_dict_keys,
                    lora_state_dict_keys=lora_state_dict_keys,
                    base_model_state_dict_keys=base_model_state_dict_keys,
                )
        else:
            validate_state_dict_for_lora(
                lora_attn_modules,
                apply_lora_to_mlp,
                apply_lora_to_output,
                full_model_state_dict_keys=full_model_state_dict_keys,
                lora_state_dict_keys=lora_state_dict_keys,
                base_model_state_dict_keys=base_model_state_dict_keys,
            )


class TestGetMergedLoRACkpt:
    def dummy_model(self):
        model = nn.Sequential(
            LoRALinear(in_dim=4, out_dim=6, rank=RANK, alpha=ALPHA),
            nn.Linear(6, 3),
        )
        model[0].lora_a.weight = nn.Parameter(
            torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        )
        model[0].lora_b.weight = nn.Parameter(
            torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        )
        model[0].weight = nn.Parameter(3 * torch.ones((6, 4)))
        return model

    def test_get_merged_lora_ckpt(self):
        dummy_model = self.dummy_model()
        merged_sd = get_merged_lora_ckpt(
            deepcopy(dummy_model.state_dict()), rank=RANK, alpha=ALPHA
        )
        expected_merged_weight = torch.Tensor(
            [
                [8.5, 10.0, 11.5, 13.0],
                [14.5, 18.0, 21.5, 25.0],
                [20.5, 26.0, 31.5, 37.0],
                [26.5, 34.0, 41.5, 49.0],
                [32.5, 42.0, 51.5, 61.0],
                [38.5, 50.0, 61.5, 73.0],
            ]
        )
        assert merged_sd.keys() == {"0.weight", "1.weight", "1.bias"}
        torch.testing.assert_close(merged_sd["0.weight"], expected_merged_weight)

        merged_model = nn.Sequential(nn.Linear(4, 6, bias=False), nn.Linear(6, 3))
        merged_model.load_state_dict(merged_sd, strict=True)

        inputs = torch.randn(2, 8, 4)
        torch.testing.assert_close(dummy_model(inputs), merged_model(inputs))
