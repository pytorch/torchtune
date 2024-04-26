# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tests.test_utils import init_weights_with_constant
from tests.torchtune.models.mistral.scripts.mistral_reference import (
    apply_rotary_emb,
    Attention,
    precompute_freqs_cis,
)
from torch import nn
from torchtune.modules import CausalSelfAttention, RotaryPositionalEmbeddings


def compare_rope(
    bsz: int, num_heads: int, embed_dim: int, seq_len: int, max_seq_len: int
) -> None:
    # make sure we have the right seed for generating outputs
    torch.manual_seed(0)

    head_dim = embed_dim // num_heads

    # generate input tensor
    x = torch.randn(bsz, seq_len, num_heads, head_dim)

    # Compute the reference tensors - mistral's implementation applies the same operation
    # to both key and query tensors at once, so we can just pass our input tensor for both
    # values and ignore the second return value.
    freq_cis = precompute_freqs_cis(dim=head_dim, end=max_seq_len * 2, theta=10000.0)
    x_out_ref, _ = apply_rotary_emb(x, x.clone(), freqs_cis=freq_cis[:seq_len])

    # Compute the tensors from current implementation
    rope_emb = RotaryPositionalEmbeddings(
        dim=head_dim, max_seq_len=max_seq_len, base=10000
    )
    x_out = rope_emb(x)

    # Validate correctness
    torch.testing.assert_close(x_out_ref, x_out, atol=1e-6, rtol=1e-5)
    print("Rope embeddings are correct")
    # value: tensor(6.4543e-05)
    print(f"x_out.mean(): {x_out.mean()}")

    # value: tensor(2165.7053)
    print(f"x_out.sum() {x_out.sum()}")

    # value: tensor(5.4546)
    print(f"x_out.max() {x_out.max()}")

    curr_pos = 10
    x_out_ref, _ = apply_rotary_emb(
        x, x.clone(), freqs_cis=freq_cis[curr_pos : curr_pos + seq_len]
    )

    x_out = rope_emb(x, input_pos=torch.arange(curr_pos, curr_pos + seq_len))

    # Validate correctness
    torch.testing.assert_close(x_out_ref, x_out, atol=1e-6, rtol=1e-5)
    print("Rope embeddings for a specific position are correct")
    # value: tensor(0.0002)
    print(f"x_out.mean(): {x_out.mean()}")

    # value: tensor(5158.3159)
    print(f"x_out.sum(): {x_out.sum()}")

    # value: tensor(5.4543)
    print(f"x_out.max(): {x_out.max()}")


def compare_attention(
    bsz: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
):
    # make sure we have the right seed for generating outputs
    torch.manual_seed(16)

    head_dim = embed_dim // num_heads

    # generate input tensor
    input_t = torch.randn(bsz, seq_len, embed_dim)
    # generate mask and frequencies tensor needed for the reference implementation
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    freq_cis = precompute_freqs_cis(dim=head_dim, end=seq_len, theta=10000.0)
    # initialize reference implementation with constant weights
    attn_ref = Attention(
        n_heads=num_heads,
        head_dim=head_dim,
        dim=embed_dim,
        n_kv_heads=num_kv_heads,
    )
    init_weights_with_constant(attn_ref, constant=0.05)

    with torch.no_grad():
        # mistral implementation expects mask [b, num_heads, seq_len, seq_len]
        attn_out_ref = attn_ref(input_t, freq_cis, mask=mask.squeeze())

    # initialise current implementation with constant weights
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=10000)
    attn = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
        k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
        pos_embeddings=rope,
        kv_cache=None,
        max_seq_len=max_seq_len,
    )

    init_weights_with_constant(attn, constant=0.05)

    with torch.no_grad():
        attn_out = attn(input_t)

    # value: tensor(-27.5074)
    print(f"attn_out.mean(): {attn_out.mean()}")

    # output tensors should be similar
    torch.testing.assert_close(attn_out, attn_out_ref, atol=1e-5, rtol=1e-3)


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

    compare_rope(
        args.bsz, args.num_heads, args.embed_dim, args.seq_len, args.max_seq_len
    )

    compare_attention(
        args.bsz,
        args.seq_len,
        args.embed_dim,
        args.num_heads,
        args.num_kv_heads,
        args.max_seq_len,
    )
