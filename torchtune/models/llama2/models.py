# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.models.llama2.tokenizer import Tokenizer
from torchtune.models.llama2.transformer import TransformerDecoder


def llama2_7b(vocab_size: int) -> TransformerDecoder:
    """
    Returns a `TransformerDecoder` architecture consistent with the
    Llama2 architecture: https://arxiv.org/abs/2307.09288.

    Args:
        vocab_size (int): vocabulary size to be used in the language modeling task.
            This is typically determined from the tokenizer and is typically the
            number of distinct tokens in the vocabulary.

    Returns:
        TransformerDecoder: A TransformerDecoder instance consistent with the Llama2 7 billion parameter architecture.
    """
    return TransformerDecoder(
        vocab_size=vocab_size,
        num_layers=32,
        num_heads=32,
        embed_dim=4096,
        max_seq_len=2048,
        norm_eps=1e-5,
    )


def llama2_tokenizer(path: str) -> Tokenizer:
    """
    Returns a `Tokenizer` instance given a pretrained tokenizer path.
    Args:
        path (str): The path to the SentencePiece model file.

    Returns:
        Tokenizer: A `Tokenizer` instance.
    """
    tokenizer = Tokenizer.from_file(path)
    # Original tokenizer has no pad_id, which causes indexing errors when batch training
    tokenizer.pad_id = 0
    return tokenizer
