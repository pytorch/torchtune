# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn, Tensor


class SDPA(nn.Module):
    """
    The core of SDPA which can be optimized and can be swapped
    out for a more efficient implemmentations.

    TODO: word the above docstring better.
    """

    def __init__(
        self,
        attention_fn,
        kv_cache,
        q_per_kv,
    ) -> None:
        super().__init__()
        self._attention_fn = attention_fn
        self.kv_cache = kv_cache
        self.q_per_kv = q_per_kv

    def kv_cache_update(
        self,
        input_pos: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        k, v = self.kv_cache.update(input_pos, k, v)
        return k, v

    def sdpa(
        self,
        q: Tensor,  # [b, s, n_h, h_d]
        k: Tensor,  # [b, s, n_kv, h_d]
        v: Tensor,  # [b, s, n_kv, h_d]
        bsz: int,
        seq_len: int,
        mask: Tensor = None,
    ) -> Tensor:
        # View + expand + reshape bring num_kv_heads to num_heads for k and v
        # to match q.

        # k: [bsz, seq_len, n_kv, 1, h_d]
        # v: [bsz, seq_len, n_kv, 1, h_d]
        k = k.view(bsz, seq_len, self.num_kv_heads, 1, self.head_dim)
        v = v.view(bsz, seq_len, self.num_kv_heads, 1, self.head_dim)

        # Expand the key and value tensors to have the same shape
        # as the query tensor by copying values across the relevant dim
        if self.num_heads != self.num_kv_heads:
            k = k.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
            v = v.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)

        # [bsz, s, n_h, h_d]
        k = k.reshape(bsz, seq_len, -1, self.head_dim)
        v = v.reshape(bsz, seq_len, -1, self.head_dim)

        # [bsz, n_h, s, h_d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        output = self._attention_fn(
            q,
            k,
            v,
            mask=mask,
            dropout_p=self.attn_dropout,
            is_causal=self.kv_cache is None and mask is None and self.is_causal,
        )
        # Reshape the output to be the same shape as the input
        return output.transpose(1, 2).contiguous().view(b, s_x, -1)
