# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import pytest
from torchtune.models.llama2 import llama2, lora_llama2

from torchtune.modules import TransformerDecoder


def llama2_small_test_ckpt(max_batch_size: Optional[int] = None) -> TransformerDecoder:
    return llama2(
        vocab_size=32_000,
        num_layers=4,
        num_heads=16,
        embed_dim=256,
        max_seq_len=2048,
        norm_eps=1e-5,
        num_kv_heads=8,
        max_batch_size=max_batch_size,
    )


def lora_llama2_small_test_ckpt(
    lora_attn_modules,
    apply_lora_to_mlp,
    lora_rank,
    lora_alpha,
    max_batch_size: Optional[int] = None,
) -> TransformerDecoder:
    return lora_llama2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        vocab_size=32_000,
        num_layers=4,
        num_heads=16,
        embed_dim=256,
        max_seq_len=2048,
        norm_eps=1e-5,
        num_kv_heads=8,
        max_batch_size=max_batch_size,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
    )


def fetch_loss_values(output) -> Dict[str, float]:
    lines = output.splitlines()
    loss_values = {}
    for line in lines:
        if "Loss:" in line:
            splits = line.split("Loss:")
            loss_value = float(splits[1].split(":")[0])
            loss_values[splits[0]] = loss_value
    return loss_values


def fetch_ckpt_model_path(ckpt) -> str:
    # TODO: same checkpoint is returned for small scale llama2
    # and lora. This should be fine as the lora adapter params
    # are initialized, but we may want to load in a lora specific
    # checkpoint.
    if "small_test_ckpt" in ckpt:
        return "/tmp/test-artifacts/small-ckpt-01242024"
    if ckpt == "llama2_7b":
        return "/tmp/test-artifacts/llama2-7b-01242024"
    raise ValueError(f"Unknown ckpt {ckpt}")


def validate_loss_values(loss_values, expected_loss_values):
    assert len(loss_values) == len(expected_loss_values)
    for key, value in loss_values.items():
        assert key in expected_loss_values
        expected_loss_value = expected_loss_values[key]
        assert value == pytest.approx(expected_loss_value, abs=0.001)
