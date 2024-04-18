# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torchtune.models.gemma._component_builders import gemma

from torchtune.modules import TransformerDecoder
from torchtune.modules.tokenizers import SentencePieceTokenizer

"""
Model builders build specific instantiations using component builders. For example
the ``gemma_2b`` model builder uses the ``gemma`` component builder.
"""


def gemma_2b() -> TransformerDecoder:
    """
    Builder for creating a Gemma 2B model initialized w/ the default 2b parameter values
    from: https://blog.google/technology/developers/gemma-open-models/

    Returns:
        TransformerDecoder: Instantiation of Gemma 2B model
    """
    return gemma(
        vocab_size=256_000,
        num_layers=18,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        embed_dim=2048,
        intermediate_dim=16384,
        max_seq_len=8192,
        attn_dropout=0.0,
        norm_eps=1e-6,
    )


def gemma_tokenizer(path: str) -> SentencePieceTokenizer:
    tokenizer = SentencePieceTokenizer(path)
    tokenizer.pad_id = 0
    return tokenizer
