# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from tests.test_utils import init_weights_with_constant
from tests.torchtune.models.mistral.scripts.mistral_reference import Transformer
from tests.torchtune.models.mistral.scripts.mistral_test_config import MistralTestConfig

from torchtune.models.mistral import mistral


def compare_decoder(
    bsz: int,
    vocab_size: int,
    seq_len: int,
    embed_dim: int,
    intermediate_dim: int,
    n_layers: int,
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
    x_input = torch.randint(low=0, high=vocab_size, size=(bsz, seq_len))

    # initialize reference implementation with constant weights
    ref_decoder = Transformer(
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=num_heads,
        head_dim=head_dim,
        dim=embed_dim,
        n_kv_heads=num_kv_heads,
        hidden_dim=intermediate_dim,
        max_seq_len=max_seq_len,
        rope_base=rope_base,
        norm_eps=norm_eps,
    )
    init_weights_with_constant(ref_decoder, constant=0.05)

    with torch.no_grad():
        decoder_out_ref = ref_decoder(x_input)

    # current implementation; initialize with constant to compare outputs
    decoder = mistral(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=n_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        intermediate_dim=intermediate_dim,
        norm_eps=norm_eps,
        rope_base=rope_base,
    )
    init_weights_with_constant(decoder, constant=0.05)

    with torch.no_grad():
        decoder_out = decoder(x_input)

    # value: torch.tensor(0.15999)
    print(f"decoder_out.mean(): {decoder_out.mean()}")
    print(f"decoder_out_ref.mean(): {decoder_out_ref.mean()}")

    torch.testing.assert_close(decoder_out, decoder_out_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare Decoder implementations")
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
        "--vocab_size",
        type=int,
        default=MistralTestConfig.VOCAB_SIZE,
        help="vocab size",
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
        "--num_layers",
        type=int,
        default=MistralTestConfig.NUM_LAYERS,
        help="number of transformer layers",
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

    compare_decoder(
        args.bsz,
        args.vocab_size,
        args.seq_len,
        args.embed_dim,
        args.intermediate_dim,
        args.num_layers,
        args.num_heads,
        args.num_kv_heads,
        args.max_seq_len,
        args.rope_base,
        args.norm_eps,
    )
