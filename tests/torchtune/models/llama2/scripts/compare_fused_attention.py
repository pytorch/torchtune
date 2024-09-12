# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from tests.test_utils import fixed_init_model
from torch import nn, Tensor
from torchtune.modules import KVCache, MultiHeadAttention, RotaryPositionalEmbeddings


# Copy-paste of fused attention for comparison
class FusedMultiHeadAttention(nn.Module):
    """Multi-headed grouped query self-attention (GQA) layer introduced
    in https://arxiv.org/pdf/2305.13245v1.pdf.

    GQA is a version of multiheaded attention (MHA) which uses fewer
    key/value heads than query heads by grouping n query heads for each
    key and value head. Multi-Query Attention is an extreme
    version where we have a single key and value head shared by all
    query heads.

    Following is an example of MHA, GQA and MQA with num_heads = 4

    (credit for the documentation:
    https://github.com/Lightning-AI/lit-gpt/blob/main/lit_gpt/config.py).


    ::

        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │         │        │                 │
        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
        ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
        │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
        └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
        ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
                MHA                    GQA                   MQA
        n_kv_heads =4          n_kv_heads=2           n_kv_heads=1

    Args:
        embed_dim (int): embedding dimension for the model
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        head_dim (int): dimension of each head, calculated by ``embed_dim`` // ``num_heads``.
        qkv_proj (nn.Module): projection layer for query, key and value.
        output_proj (nn.Module): projection layer for output.
        pos_embeddings (nn.Module): positional embeddings layer, e.g. RotaryPositionalEmbeddings.
        kv_cache (Optional[KVCache]): KVCache object used to cache key and value.
            If not specified, then no caching is used.
        max_seq_len (int): maximum sequence length supported by the model.
            This is needed to compute the RoPE Cache. Default: 4096.
        attn_dropout (float): dropout value passed onto the
            scaled_dot_product_attention function. This argument is ignored if the
            self.training is False. Default value is 0.0.

    Raises:
        ValueError: If `num_heads` % `num_kv_heads` != 0
        ValueError: If `embed_dim` % `num_heads` != 0
        ValueError: If `attn_dropout` < 0 or > 1
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qkv_proj: nn.Module,
        output_proj: nn.Module,
        pos_embeddings: nn.Module,
        kv_cache: Optional[KVCache] = None,
        max_seq_len: int = 4096,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f"attn_dropout ({embed_dim}) must be between 0.0 and 1.0")

        # Set attributes
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Set layers
        self.kv_cache = kv_cache
        self.qkv_proj = qkv_proj
        self.output_proj = output_proj
        self.pos_embeddings = pos_embeddings

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        curr_pos: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[torch.Tensor]): boolean mask, defaults to None.
            curr_pos (int): current position in the sequence, defaults to 0.

        Returns:
            Tensor: output tensor with attention applied

        Raises:
            ValueError: if seq_len of x is bigger than max_seq_len

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - n_kv: num kv heads
            - d: embed dim
            - h_d: head dim
            - qkv_d: qkv_dim computed as (n_h + 2 * n_kv) * h_d

        TODO:
            - Return the attention weights
            - Make application of positional embeddings optional
        """

        # input has shape [b, s, d]
        bsz, seq_len, _ = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        # qkv has shape [b, s, qkv_d]
        qkv = self.qkv_proj(x)

        # number of queries per key/value
        q_per_kv = self.num_heads // self.num_kv_heads

        # Each key and value either has a single query (MHA)
        # or q_per_kv queries (MQA/GQA). total_qkv will be 3
        # for MHA
        total_qkv = q_per_kv + 2

        # decompose the last dimension into n_kv x total_qkv, h_d
        qkv = qkv.view(bsz, seq_len, self.num_kv_heads, total_qkv, self.head_dim)

        # create the q,k and v tensors by splitting qkv
        # q: [b, s, n_kv, q_per_kv, h_d]
        # k: [b, s, n_kv, 1, h_d]
        # v: [b, s, n_kv, 1, h_d]
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=3)

        # if needed, expand the key and value tensors to have the same shape
        # as the query tensor by copying values across the relevant dim
        if self.num_heads != self.num_kv_heads:
            k = k.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
            v = v.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)

        # llama2 applies the RoPE embeddings on tensors with shape
        # [b, s, n_h, h_d]
        # Reshape the tensors before we apply RoPE
        q = q.reshape(bsz, seq_len, -1, self.head_dim)
        k = k.reshape(bsz, seq_len, -1, self.head_dim)
        v = v.reshape(bsz, seq_len, -1, self.head_dim)

        # Apply positional embeddings
        q = self.pos_embeddings(q, curr_pos)
        k = self.pos_embeddings(k, curr_pos)

        # Update key-value cache
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(
                bsz=bsz, seq_len=seq_len, curr_pos=curr_pos, k_val=k, v_val=v
            )

        # [b, n_h, s, h_d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash attention from https://pytorch.org/blog/accelerating-large-language-models/
        output = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout,
            is_causal=self.kv_cache is None,
        )

        # reshape the output to be the same shape as the input
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.output_proj(output)


def map_state_dict(
    sd,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
):
    mapped_sd = {k: v for k, v in sd.items() if "qkv_proj" not in k}
    q_per_kv = num_heads // num_kv_heads
    slice_size = q_per_kv + 2
    ind = range(head_dim * num_kv_heads * slice_size)
    qkv = sd["qkv_proj.weight"]
    q_ind = list(filter(lambda x: (x // head_dim) % slice_size < slice_size - 2, ind))
    k_ind = list(filter(lambda x: (x // head_dim) % slice_size == slice_size - 2, ind))
    v_ind = list(filter(lambda x: (x // head_dim) % slice_size == slice_size - 1, ind))
    q = qkv.index_select(0, torch.tensor(q_ind))
    k = qkv.index_select(0, torch.tensor(k_ind))
    v = qkv.index_select(0, torch.tensor(v_ind))
    mapped_sd["q_proj.weight"] = q
    mapped_sd["k_proj.weight"] = k
    mapped_sd["v_proj.weight"] = v
    return mapped_sd


def _get_mask(inpt: torch.Tensor) -> torch.Tensor:
    seq_len = inpt.shape[1]
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=inpt.device)
    mask = torch.triu(mask, diagonal=1).type_as(inpt)
    return mask


def compare_attn(
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    use_kv_cache: bool,
):

    torch.manual_seed(16)
    inputs = torch.randn(4, 2048, 4096)

    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim

    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
    if use_kv_cache:
        kv_cache = KVCache(
            batch_size=4,
            max_seq_len=max_seq_len,
            n_kv_heads=num_heads,
            head_dim=head_dim,
        )
    else:
        kv_cache = None

    attn_ref = FusedMultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qkv_proj=nn.Linear(embed_dim, qkv_dim, bias=False),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
        pos_embeddings=rope,
        kv_cache=kv_cache,
        max_seq_len=max_seq_len,
    )
    fixed_init_model(attn_ref)
    attn_ref.eval()

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
        kv_cache=kv_cache,
        max_seq_len=max_seq_len,
    )
    mapped_sd = map_state_dict(attn_ref.state_dict(), head_dim, num_heads, num_kv_heads)
    attn.load_state_dict(mapped_sd)

    # Compare fused and non-fused with remapped state dict
    with torch.no_grad():
        if use_kv_cache:
            mask = _get_mask(inputs)
            out_ref = attn_ref(inputs, mask, curr_pos=0)
            out = attn_ref(inputs, mask, curr_pos=0)
        else:
            out_ref = attn_ref(inputs)
            out = attn(inputs)
    print(
        "These values should match the original unit test", out_ref.mean(), out.mean()
    )
    torch.testing.assert_close(out_ref, out, atol=1e-8, rtol=1e-3)

    # Determine the new value with fixed initialization
    fixed_init_model(attn)
    with torch.no_grad():
        if use_kv_cache:
            new_out = attn(inputs, mask, curr_pos=0)
        else:
            new_out = attn(inputs)
    print(f"New unit test value: {new_out.mean()}")


if __name__ == "__main__":

    # compare mha
    mha = {
        "num_heads": 32,
        "embed_dim": 4096,
        "max_seq_len": 4096,
        "num_kv_heads": None,
        "use_kv_cache": False,
    }
    mqa = {
        "num_heads": 32,
        "embed_dim": 4096,
        "max_seq_len": 4096,
        "num_kv_heads": 1,
        "use_kv_cache": False,
    }
    gqa = {
        "num_heads": 32,
        "embed_dim": 4096,
        "max_seq_len": 4096,
        "num_kv_heads": 8,
        "use_kv_cache": False,
    }
    mha_kv = {
        "num_heads": 32,
        "embed_dim": 4096,
        "max_seq_len": 4096,
        "num_kv_heads": None,
        "use_kv_cache": True,
    }
    mqa_kv = {
        "num_heads": 32,
        "embed_dim": 4096,
        "max_seq_len": 4096,
        "num_kv_heads": 1,
        "use_kv_cache": True,
    }
    gqa_kv = {
        "num_heads": 32,
        "embed_dim": 4096,
        "max_seq_len": 4096,
        "num_kv_heads": 8,
        "use_kv_cache": True,
    }
    test_cases = {
        "mha": mha,
        "mqa": mqa,
        "gqa": gqa,
        "mha_kv": mha_kv,
        "mqa_kv": mqa_kv,
        "gqa_kv": gqa_kv,
    }

    for test_case, params in test_cases.items():
        print(f"For test case {test_case}")
        compare_attn(**params)
