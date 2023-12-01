# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torch import nn

from torchtune.llm.llama2.transformer import TransformerDecoder

from tests.llm.llama2.scripts.compare_attention import precompute_freqs_cis
from tests.llm.llama2.scripts.compare_decoder_layer import RMSNorm, TransformerBlock

from tests.test_utils import init_weights_with_constant

"""
Reference implementation of Transformer from:
https://github.com/facebookresearch/llama/blob/main/llama/model.py#L413

Replicating code here to minimize dependencies. The code is modified to
include params for the constructor and remove start_pos (not supported).
"""

def compare_decoder(
    bsz: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
    vocab_size: int,
    num_layers: int,
) -> None:

    # make sure we have the right seed for generating outputs
    # this should match up the seed value set in the corresponding
    # unit test
    torch.manual_seed(16)

    head_dim = embed_dim // num_heads

    # generate input tensor used by both implementations
    x_input = torch.randint(low=0, high=vocab_size, size=(bsz, seq_len))

    # reference implementation; initialize with constant to compare outputs
    decoder_ref = Transformer(
        vocab_size=vocab_size,
        dim=embed_dim,
        n_layers=num_layers,
        n_heads=num_heads,
        max_seq_len=max_seq_len,
        n_kv_heads=num_kv_heads,
    )
    init_weights_with_constant(decoder_ref, constant=0.2)

    with torch.no_grad():
        output_ref = decoder_ref(x_input)

    # current implementation; initialize with constant to compare outputs
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        num_kv_heads=num_kv_heads,
    )
    init_weights_with_constant(decoder, constant=0.2)

    with torch.no_grad():
        output = decoder(x_input)

    # value: tensor(163.8399)
    print(output.mean())

    assert torch.allclose(output_ref, output, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare Attention implementations")
    parser.add_argument("--bsz", type=int, default=4, help="Batch size of input tensor")
    parser.add_argument(
        "--seq_len", type=int, default=2048, help="input sequence length"
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=4096,
        help="Embedding dimension used to compute the dim for RopE",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=32,
        help="Number of heads in the attention layer",
    )
    parser.add_argument(
        "--num_kv_heads",
        type=int,
        default=8,
        help="Number of key/value heads in the attention layer",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=4096, help="max sequence length"
    )
    parser.add_argument("--vocab_size", type=int, default=1024, help="vocab size")
    parser.add_argument(
        "--num_layers", type=int, default=4, help="number of transformer layers"
    )

    args = parser.parse_args()

    compare_decoder(
        args.bsz,
        args.seq_len,
        args.embed_dim,
        args.num_heads,
        args.num_kv_heads,
        args.max_seq_len,
        args.vocab_size,
        args.num_layers,
    )
