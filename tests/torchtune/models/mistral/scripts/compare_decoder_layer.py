# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from tests.test_utils import init_weights_with_constant
from tests.torchtune.models.mistral.scripts.mistral_reference import (
    precompute_freqs_cis,
    TransformerBlock,
)
from tests.torchtune.models.mistral.scripts.mistral_test_config import MistralTestConfig

from torch import nn

from torchtune.models.mistral._component_builders import mistral_mlp

from torchtune.modules import (
    CausalSelfAttention,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TransformerDecoderLayer,
)


def compare_decoder_layer(
    bsz: int,
    seq_len: int,
    embed_dim: int,
    intermediate_dim: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
    rope_base: int,
    norm_eps: float,
) -> None:
    # make sure we have the right seed for generating outputs
    # this should match up the seed value set in the corresponding
    # unit test
    torch.manual_seed(MistralTestConfig.SEED)

    head_dim = embed_dim // num_heads

    # generate input tensor used by both implementations
    input_t = torch.randn(bsz, seq_len, embed_dim)
    head_dim = embed_dim // num_heads

    # generate input tensor
    input_t = torch.randn(bsz, seq_len, embed_dim)
    # generate mask and frequencies tensor needed for the reference implementation
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    freq_cis = precompute_freqs_cis(dim=head_dim, end=seq_len, theta=float(rope_base))
    # initialize reference implementation with constant weights
    ref_decoder_layer = TransformerBlock(
        n_heads=num_heads,
        head_dim=head_dim,
        dim=embed_dim,
        n_kv_heads=num_kv_heads,
        hidden_dim=intermediate_dim,
        norm_eps=norm_eps,
    )
    init_weights_with_constant(ref_decoder_layer, constant=0.05)

    with torch.no_grad():
        # mistral implementation expects mask [b, num_heads, seq_len, seq_len]
        decoder_out_ref = ref_decoder_layer(
            x=input_t, freqs_cis=freq_cis, mask=mask.squeeze()
        )

    # current implementation; initialize with constant to compare outputs
    norm_eps = 1e-5
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    rope = RotaryPositionalEmbeddings(
        dim=head_dim, max_seq_len=max_seq_len, base=rope_base
    )
    self_attn = CausalSelfAttention(
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
    mlp = mistral_mlp(dim=embed_dim, hidden_dim=intermediate_dim)
    decoder_layer = TransformerDecoderLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
    )
    init_weights_with_constant(decoder_layer, constant=0.05)

    with torch.no_grad():
        decoder_layer_out = decoder_layer(input_t)

    # value: torch.tensor(-0.00133)
    print(f"decoder_out.mean(): {decoder_layer_out.mean()}")

    torch.testing.assert_close(decoder_layer_out, decoder_out_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare Attention implementations")
    parser.add_argument(
        "--bsz",
        type=int,
        default=MistralTestConfig.BSZ,
        help="Batch size of input tensor",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=MistralTestConfig.SEQ_LEN,
        help="input sequence length",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=MistralTestConfig.EMBED_DIM,
        help="Embedding dimension used to compute the dim for RopE",
    )
    parser.add_argument(
        "--intermediate_dim",
        type=int,
        default=MistralTestConfig.INTERMEDIATE_DIM,
        help="Intermediate dimension for MLP",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=MistralTestConfig.NUM_HEADS,
        help="Number of heads in the attention layer",
    )
    parser.add_argument(
        "--num_kv_heads",
        type=int,
        default=MistralTestConfig.NUM_KV_HEADS,
        help="Number of key/value heads in the attention layer",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=MistralTestConfig.MAX_SEQ_LEN,
        help="max sequence length",
    )
    parser.add_argument(
        "--norm_eps",
        type=float,
        default=MistralTestConfig.NORM_EPS,
        help="RMSNorm epsilon",
    )
    parser.add_argument(
        "--rope_base",
        type=float,
        default=MistralTestConfig.ROPE_BASE,
        help="Base for the rotary positional embeddings",
    )
    args = parser.parse_args()

    compare_decoder_layer(
        args.bsz,
        args.seq_len,
        args.embed_dim,
        args.intermediate_dim,
        args.num_heads,
        args.num_kv_heads,
        args.max_seq_len,
        args.rope_base,
        args.norm_eps,
    )
