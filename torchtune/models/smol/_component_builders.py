# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torchtune.modules import (
    TransformerDecoder,
)
from torchtune.models.llama3_2._component_builders import llama3_2

"""
Component builders for SmolLM 2. It is based on LLaMA architecture.

https://huggingface.co/HuggingFaceTB/SmolLM2-135M/

SmolLM2 is a family of compact language models available in three size: 135M, 360M, 
and 1.7B parameters. They are capable of solving a wide range of tasks while being 
lightweight enough to run on-device. More details in our paper: https://arxiv.org/abs/2502.02737v1
"""


def smollm2(
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int = 8192,
    vocab_size: int = 49152,
    attn_dropout: float = 0.0,
    rope_base: int = 100000,
    norm_eps: float = 1e-5,
    scale_factor: int = 32,
    tie_word_embeddings: bool = True,
) -> TransformerDecoder:
    return llama3_2(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
        rope_base=rope_base,
        intermediate_dim=intermediate_dim,
        norm_eps=norm_eps,
        scale_factor=scale_factor,
        tie_word_embeddings=tie_word_embeddings,
    )
