# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from torch import nn

"""
Reference mistral implementation from the official repo:
https://github.com/mistralai/mistral-src/blob/main/one_file_ref.py

Components are copied here with minimal modifications.
"""


"""
Reference implementation of Attention from:
https://github.com/mistralai/mistral-src/blob/main/one_file_ref.py

Note, there's another implementation in the same repo which uses xformers for attention:
https://github.com/mistralai/mistral-src/blob/8598cf582091a596671be31990448e0620017851/mistral/model.py#L60

The implementation for this test uses `one_file_ref.py` since the xformers attention implementation
expects the input `[b, s, ...]` to be flattened `[b * s, ...]` which makes comparison difficult.

Replicating code here to minimize dependencies. The code is modified to
remove dependencies from xformers and features like KV Caching.
"""


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=2)
    values = torch.repeat_interleave(values, repeats=repeats, dim=2)
    return keys, values


class Attention(nn.Module):
    def __init__(self, n_heads: int, head_dim: int, dim: int, n_kv_heads: int):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = head_dim**-0.5

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # removed positions as it was only used for cache retrieval
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        key, value = repeat_kv(xk, xv, self.repeats)

        query = xq.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # scores : [bsz, n_heads, seqlen | 1, seqlen]
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scale
        print(scores.mean())
        if mask is not None:
            scores += mask[None, None, ...]
        print(scores.mean())
        scores = scores.float()
        scores = nn.functional.softmax(scores, dim=-1).type_as(query)
        output = torch.matmul(scores, value)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


"""
Reference implementation of RoPE from:
https://github.com/mistralai/mistral-src/blob/8598cf582091a596671be31990448e0620017851/one_file_ref.py#L47

The original code structures this as stand-alone functions instead of
a class. Replicating code here to minimize dependencies.
"""


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


"""
Reference impementation of FeedForward from:
https://github.com/mistralai/mistral-src/blob/8598cf582091a596671be31990448e0620017851/one_file_ref.py#L152

The original code structures this as stand-alone functions in
`torchtune.models.mistral._component_builders.mistral_mlp` instead of
a standalone class.
"""


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


"""
Reference implementation of TransformerBlock from:
https://github.com/mistralai/mistral-src/blob/8598cf582091a596671be31990448e0620017851/one_file_ref.py#L190
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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        dim: int,
        n_kv_heads: int,
        hidden_dim: int,
        norm_eps: float,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attention = Attention(
            n_heads=n_heads, head_dim=head_dim, dim=dim, n_kv_heads=n_kv_heads
        )
        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim)
        self.attention_norm = RMSNorm(dim=dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, mask)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


"""
Reference implementation of Transformer from:
https://github.com/mistralai/mistral-src/blob/8598cf582091a596671be31990448e0620017851/one_file_ref.py#L217
"""


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        dim: int,
        n_kv_heads: int,
        hidden_dim: int,
        max_seq_len: int,
        rope_base: int,
        norm_eps: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        assert self.vocab_size > 0

        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    n_heads=n_heads,
                    head_dim=head_dim,
                    dim=dim,
                    n_kv_heads=n_kv_heads,
                    hidden_dim=hidden_dim,
                    norm_eps=norm_eps,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = RMSNorm(dim, eps=norm_eps)

        self.output = nn.Linear(dim, vocab_size, bias=False)

        # our RoPE implementation is a bit different from the reference:
        # mistral hardcodes max_seq_len and uses a `positions` argument
        # in forward to index `freqs_cis` for the current sequence length
        # before using it in the attention layer.

        self.freqs_cis = precompute_freqs_cis(
            head_dim, max_seq_len * 2, theta=rope_base
        )  # removed .to("cuda")

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        _, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[positions]
        mask: Optional[torch.Tensor] = None
        if input_ids.shape[1] > 1:
            seqlen = input_ids.shape[1]
            tensor = torch.full(
                (seqlen, seqlen),
                dtype=h.dtype,
                fill_value=1,
                device=h.device,
            )
            mask = torch.tril(tensor, diagonal=0).to(h.dtype)
            # removed mask banding
            mask = torch.triu(mask, diagonal=-1)  # setting sliding window to 1
            mask = torch.log(mask)
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        return self.output(self.norm(h)).float()
