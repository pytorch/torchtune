# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.models.llama2.tokenizer import Tokenizer
from torchtune.models.llama2.transformer import TransformerDecoder


def llama2_7b(vocab_size: int) -> TransformerDecoder:
    model = TransformerDecoder(
        vocab_size=vocab_size,
        num_layers=32,
        num_heads=32,
        embed_dim=4096,
        max_seq_len=2048,
        norm_eps=1e-5,
    )
    return model


def llama2_tokenizer(path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(path)
    # Original tokenizer has no pad_id, which causes indexing errors when batch training
    tokenizer.pad_id = 0
    return tokenizer
