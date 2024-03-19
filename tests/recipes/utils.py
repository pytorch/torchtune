# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import pytest

import torch
from tests.test_utils import get_assets_path
from torch.utils.data import Dataset
from torchtune.models.llama2 import llama2, lora_llama2

from torchtune.modules import TransformerDecoder

_ASSETS = get_assets_path()


class DummyDataset(Dataset):
    def __init__(self, **kwargs):
        self._data = torch.LongTensor(
            [
                [0, 2, 4, 2, 5, 6, 7, 8, 9, 1, 2, 4, 3, 3, 5, 6, 8, 2, 1, 1],
                [1, 2, 5, 6, 7, 8, 2, 3, 1, 9, 9, 9, 5, 6, 7, 0, 0, 0, 1, 2],
                [5, 6, 8, 2, 1, 0, 3, 4, 0, 0, 0, 2, 4, 7, 8, 8, 2, 2, 1, 0],
                [4, 6, 7, 1, 0, 2, 0, 2, 0, 2, 3, 9, 9, 9, 7, 5, 1, 8, 4, 1],
            ]
        )
        self._labels = torch.LongTensor(
            [
                [2, 6, 7, 8, 2, 2, 1, 0, 0, 1],
                [1, 2, 5, 6, 7, 8, 2, 3, 1, 9],
                [6, 1, 1, 2, 5, 0, 9, 0, 2, 1],
                [5, 8, 6, 0, 2, 0, 0, 3, 2, 1],
            ]
        )

    def __getitem__(self, index):
        return (self._data[index], self._labels[index])

    def __len__(self):
        return len(self._data)


def llama2_tiny_test_ckpt(max_batch_size: Optional[int] = None) -> TransformerDecoder:
    return llama2(
        vocab_size=100,
        num_layers=2,
        num_heads=4,
        embed_dim=64,
        max_seq_len=64,
        norm_eps=1e-5,
        num_kv_heads=2,
        max_batch_size=max_batch_size,
    )


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
    apply_lora_to_output,
    lora_rank,
    lora_alpha,
    max_batch_size: Optional[int] = None,
) -> TransformerDecoder:
    return lora_llama2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
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
    if ckpt == "small_test_ckpt_tune":
        return "/tmp/test-artifacts/small-ckpt-tune-03082024.pt"
    if ckpt == "small_test_ckpt_meta":
        return "/tmp/test-artifacts/small-ckpt-meta-03082024.pt"
    if ckpt == "small_test_ckpt_hf":
        return "/tmp/test-artifacts/small-ckpt-hf-03082024.pt"
    if "llama2_7b" in ckpt:
        return "/tmp/test-artifacts/llama2-7b-torchtune.pt"
    if "tiny_test_ckpt" in ckpt:
        return _ASSETS / "tiny_llama2_checkpoint.pt"
    raise ValueError(f"Unknown ckpt {ckpt}")


def validate_loss_values(loss_values, expected_loss_values):
    assert len(loss_values) == len(expected_loss_values)
    for key, value in loss_values.items():
        assert key in expected_loss_values
        expected_loss_value = expected_loss_values[key]
        assert value == pytest.approx(expected_loss_value, abs=0.001)
