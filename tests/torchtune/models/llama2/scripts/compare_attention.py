# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from torch import nn

from torchtune.modules import MultiHeadAttention, RotaryPositionalEmbeddings


"""
Reference implementation of Attention from:
https://github.com/facebookresearch/llama/blob/main/llama/model.py#L176

Replicating code here to minimize dependencies. The code is modified to
remove dependencies like FAIRSCale and features like KV Caching.
The latter is not supported by the current implementation.
"""


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, n_heads: int, n_kv_heads: int, dim: int):
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_heads = n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // n_heads
        self.dim = dim

        self.wq = nn.Linear(self.dim, self.dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = nn.functional.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        return output


"""
Reference implementation of RoPE from:
https://github.com/facebookresearch/llama/blob/main/llama/model.py#L80

The original code structures this as stand-alone functions instead of
a class. Replicating code here to minimize dependencies.
"""


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (
        x.shape[1],
        x.shape[-1],
    ), f"{freqs_cis.shape} does not match {x.shape[1], x.shape[-1]}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


def compare_rope(
    bsz: int, num_heads: int, embed_dim: int, seq_len: int, max_seq_len: int
) -> None:

    # make sure we have the right seed for generating outputs
    torch.manual_seed(0)

    head_dim = embed_dim // num_heads

    # generate input tensor
    x = torch.randn(bsz, seq_len, num_heads, head_dim)

    # Compute the reference tensors
    freq_cis = precompute_freqs_cis(dim=head_dim, end=max_seq_len * 2)
    x_out_ref = apply_rotary_emb(x, freqs_cis=freq_cis[:seq_len])

    # Compute the tensors from current implementation
    rope_emb = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
    x_out = rope_emb(x)

    # Validate correctness
    torch.testing.assert_close(x_out_ref, x_out, atol=1e-6, rtol=1e-5)

    # value: tensor(6.4543e-05)
    print(x_out.mean())

    # value: tensor(2165.7053)
    print(x_out.sum())

    # value: tensor(5.4546)
    print(x_out.max())

    curr_pos = 10
    x_out_ref = apply_rotary_emb(x, freqs_cis=freq_cis[curr_pos : curr_pos + seq_len])
    x_out = rope_emb(x, curr_pos=10)

    # Validate correctness
    torch.testing.assert_close(x_out_ref, x_out, atol=1e-6, rtol=1e-5)

    # value: tensor(0.0002)
    print(x_out.mean())

    # value: tensor(5158.3159)
    print(x_out.sum())

    # value: tensor(5.4543)
    print(x_out.max())


def compare_attention(
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
    attn_ref = Attention(n_heads=num_heads, n_kv_heads=num_kv_heads, dim=embed_dim)
    for p in attn_ref.parameters():
        nn.init.constant_(p, 0.05)

    with torch.no_grad():
        attn_out_ref = attn_ref(input_t, freq_cis, mask)

    # current implementation; initialize with constant to compare outputs
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
    attn = MultiHeadAttention(
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
        attn_dropout=0.0,
    )
    for p in attn.parameters():
        nn.init.constant_(p, 0.05)

    with torch.no_grad():
        attn_out = attn(input_t)

    # value: tensor(-27.5074)
    print(attn_out.mean())

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
