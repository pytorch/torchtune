# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from tests.test_utils import init_weights_with_constant

from tests.torchtune.models.llama2.scripts.compare_attention import (
    Attention,
    precompute_freqs_cis,
)
from tests.torchtune.models.llama2.scripts.compare_feed_forward import FeedForwardRef

from torch import nn

from torchtune.modules.transformer import TransformerDecoderLayer


"""
Reference implementation of RMSNorm from:
https://github.com/facebookresearch/llama/blob/main/llama/model.py#L34

Replicating code here to minimize dependencies. The code is modified to
include params for the constructor and remove start_pos (not supported).
"""


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


"""
Reference implementation of Transformer Decoder Layer from:
https://github.com/facebookresearch/llama/blob/main/llama/model.py#L351

Replicating code here to minimize dependencies. The code is modified to
include params for the constructor and remove start_pos (not supported).
"""


class TransformerBlock(nn.Module):
    def __init__(self, n_heads: int, dim: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        # self.head_dim = args.dim // args.n_heads
        self.attention = Attention(n_heads=n_heads, n_kv_heads=n_kv_heads, dim=dim)
        self.feed_forward = FeedForwardRef(dim=dim, hidden_dim=4 * dim)
        self.attention_norm = RMSNorm(dim=dim)
        self.ffn_norm = RMSNorm(dim=dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        norm_out = self.attention_norm(x)
        attn_out = self.attention.forward(norm_out, freqs_cis, mask)
        h = x + attn_out
        ffn_norm_out = self.ffn_norm(h)
        mlp_out = self.feed_forward.forward(ffn_norm_out)
        out = h + mlp_out
        return out


def compare_decoder_layer(
    bsz: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
) -> None:
    # make sure we have the right seed for generating outputs
    # this should match up the seed value set in the corresponding
    # unit test
    torch.manual_seed(16)

    head_dim = embed_dim // num_heads

    # generate input tensor used by both implementations
    input_t = torch.randn(bsz, seq_len, embed_dim)

    # generate mask and frequencies tensor needed for the reference
    # implementation
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    freq_cis = precompute_freqs_cis(dim=head_dim, end=seq_len)

    # reference implementation; initialize with constant to compare outputs
    transformer_block = TransformerBlock(
        n_heads=num_heads, n_kv_heads=num_kv_heads, dim=embed_dim
    )
    init_weights_with_constant(transformer_block, constant=0.05)

    with torch.no_grad():
        block_out = transformer_block(x=input_t, freqs_cis=freq_cis, mask=mask)

    # current implementation; initialize with constant to compare outputs
    transformer_layer = TransformerDecoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
    )
    init_weights_with_constant(transformer_layer, constant=0.05)

    with torch.no_grad():
        layer_out = transformer_layer(input_t)

    # value: torch.tensor(18261.0156)
    print(layer_out.mean())

    assert torch.allclose(block_out, layer_out, atol=1e-2, rtol=1e-2)


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

    args = parser.parse_args()

    compare_decoder_layer(
        args.bsz,
        args.seq_len,
        args.embed_dim,
        args.num_heads,
        args.num_kv_heads,
        args.max_seq_len,
    )
