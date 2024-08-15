# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List

import torch

from tests.test_utils import fixed_init_model

from torch import nn

from torchtune.models.llama2._lora_llama2_builders import _lora_llama_self_attention
from torchtune.modules import KVCache, MultiHeadAttention, RotaryPositionalEmbeddings

try:
    from peft import inject_adapter_in_model, LoraConfig
except:
    raise ImportError("Must have peft installed to run this comparison script")


def compare_lora_attention(
    bsz: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
    lora_modules: List[str],
    lora_rank: int,
    lora_alpha: float,
) -> None:

    # make sure we have the right seed for generating outputs
    # this should match up the seed value set in the corresponding
    # unit test
    torch.manual_seed(16)

    # generate input tensor used by both implementations
    x = torch.randn(bsz, seq_len, embed_dim)

    # Our implementation
    lora_llama_attn = _lora_llama_self_attention(
        lora_modules=lora_modules,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    fixed_init_model(lora_llama_attn)

    with torch.no_grad():
        out = lora_llama_attn(x)

    batch_size = None
    attn_dropout = 0.0
    # Reference implementation: wrap our native causal self-attention with PEFT LoRAConfig
    # Copy-pasted from llama2.py
    # https://github.com/pytorch/torchtune/blob/e983194629d7f093257225dafb7cbc4e46505cc8/torchtune/models/llama2.py#L88-L114
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    kv_cache = (
        KVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            n_kv_heads=num_heads,
            head_dim=head_dim,
        )
        if batch_size is not None
        else None
    )
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
    llama_attn_ref = MultiHeadAttention(
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
        attn_dropout=attn_dropout,
    )
    lora_config_ref = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        r=lora_rank,
        bias="none",
        target_modules=lora_modules,
    )

    lora_llama_attn_ref = inject_adapter_in_model(lora_config_ref, llama_attn_ref)

    all_keys = ["q_proj", "k_proj", "v_proj", "output_proj"]

    mapped_sd = {}
    for key in all_keys:
        if key in lora_modules:
            mapped_sd[f"{key}.base_layer.weight"] = lora_llama_attn.state_dict()[
                f"{key}.weight"
            ]
            mapped_sd[f"{key}.lora_A.default.weight"] = lora_llama_attn.state_dict()[
                f"{key}.lora_a.weight"
            ]
            mapped_sd[f"{key}.lora_B.default.weight"] = lora_llama_attn.state_dict()[
                f"{key}.lora_b.weight"
            ]
        else:
            mapped_sd[f"{key}.weight"] = lora_llama_attn.state_dict()[f"{key}.weight"]

    lora_llama_attn_ref.load_state_dict(mapped_sd)

    with torch.no_grad():
        out_ref = lora_llama_attn_ref(x)

    print(lora_modules, out.mean(), out_ref.mean(), out.shape, out_ref.shape)

    # output tensors should be similar
    torch.testing.assert_close(out, out_ref, atol=1e-5, rtol=1e-3)


if __name__ == "__main__":
    test_cases = [
        ["q_proj", "v_proj"],
        ["q_proj", "k_proj", "v_proj", "output_proj"],
        ["k_proj"],
    ]
    for lora_modules in test_cases:
        compare_lora_attention(
            bsz=2,
            seq_len=32,
            embed_dim=64,
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=64,
            lora_modules=lora_modules,
            lora_rank=4,
            lora_alpha=1.0,
        )
