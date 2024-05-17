# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from tests.torchtune.models.llama2.scripts.compare_attention import precompute_freqs_cis
from tests.torchtune.models.llama2.scripts.compare_decoder_layer import (
    RMSNormRef,
    TransformerBlock,
)

from torch import nn

from torchtune.models.llama2 import llama2

"""
Reference implementation of Transformer from:
https://github.com/facebookresearch/llama/blob/main/llama/model.py#L413

Replicating code here to minimize dependencies. The code is modified to
include params for the constructor and remove start_pos (not supported).
"""

# TODO: Move this to standalone ref implementation
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int,
        n_kv_heads: int,
    ):

        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        self.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                TransformerBlock(n_heads=n_heads, dim=dim, n_kv_heads=n_kv_heads)
            )

        self.norm = RMSNormRef(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output


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
    for p in model.parameters():
        nn.init.constant_(decoder_ref, 0.2)

    with torch.no_grad():
        output_ref = decoder_ref(x_input)

    # current implementation; initialize with constant to compare outputs
    decoder = llama2(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        num_kv_heads=num_kv_heads,
    )
    for p in model.parameters():
        nn.init.constant_(decoder, 0.2)

    with torch.no_grad():
        output = decoder(x_input)

    # value: tensor(20.4800)
    print(output.mean())

    torch.testing.assert_close(output_ref, output, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare Attention implementations")
    parser.add_argument("--bsz", type=int, default=4, help="Batch size of input tensor")
    parser.add_argument(
        "--seq_len", type=int, default=512, help="input sequence length"
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=512,
        help="Embedding dimension used to compute the dim for RopE",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of heads in the attention layer",
    )
    parser.add_argument(
        "--num_kv_heads",
        type=int,
        default=8,
        help="Number of key/value heads in the attention layer",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="max sequence length"
    )
    parser.add_argument("--vocab_size", type=int, default=256, help="vocab size")
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
