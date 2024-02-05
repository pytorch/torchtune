# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import recipes.alpaca_generate as alpaca_generate
from torchtune import models
from torchtune.models.llama2 import llama2
from torchtune.modules import TransformerDecoder


def small_test_ckpt(max_batch_size: Optional[int] = None) -> TransformerDecoder:
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


models.ALL_MODELS["small_test_ckpt"] = small_test_ckpt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAlpacaGenerateRecipe:
    def _fetch_ckpt_model_path(self, ckpt) -> str:
        if ckpt == "small_test_ckpt":
            return "/tmp/test-artifacts/small-ckpt-01242024"
        if ckpt == "llama2_7b":
            return "/tmp/test-artifacts/llama2-7b-01242024"
        raise ValueError(f"Unknown ckpt {ckpt}")

    def test_alpaca_generate(self, capsys, pytestconfig):
        large_scale = pytestconfig.getoption("--large-scale")
        ckpt = "llama2_7b" if large_scale else "small_test_ckpt"

        kwargs_values = {
            "model": ckpt,
            "model_checkpoint": self._fetch_ckpt_model_path(ckpt),
            "tokenizer": "llama2_tokenizer",
            "tokenizer_checkpoint": "/tmp/test-artifacts/tokenizer.model",
            "instruction": "Answer the question.",
            "input": "What is some cool music from the 1920s?",
            "max_gen_len": 64,
        }
        alpaca_generate.recipe(**kwargs_values)
