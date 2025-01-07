# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.models.t5._component_builders import t5_encoder
from torchtune.models.t5._encoder import T5Encoder
from torchtune.models.t5._tokenizer import T5Tokenizer


def t5_v1_1_xxl_encoder(max_seq_len: int = 512) -> T5Encoder:
    """
    Builder for the T5 v1.1 XXL (11B parameters) encoder.

    T5 paper: https://arxiv.org/abs/1910.10683

    1.1 release:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511

    Args:
        max_seq_len (int): The maximum sequence length (context length) of the model.
            Default: 512

    Returns:
        T5Encoder: Instantiation of the T5 encoder
    """
    return t5_encoder(
        embed_dim=4096,
        mlp_dim=10240,
        num_heads=64,
        head_dim=64,
        num_layers=24,
        rel_pos_num_buckets=32,
        rel_pos_max_dist=128,
        vocab_size=32128,
        norm_eps=1e-6,
        max_seq_len=max_seq_len,
    )


def t5_tokenizer(path: str, max_seq_len: int = 512, truncate: bool = True):
    """
    Builder for the T5 tokenizer.

    Args:
        path (str): the path to the T5 sentencepiece tokenizer file
        max_seq_len (int): the context length
        truncate (bool): whether to truncate the token sequence when longer than max_seq_len

    Returns:
        T5Tokenizer: Instantiation of the T5 tokenizer
    """
    return T5Tokenizer(path, max_seq_len=max_seq_len, truncate=truncate)
