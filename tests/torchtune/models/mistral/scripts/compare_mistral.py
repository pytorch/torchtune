# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from tests.test_utils import fixed_init_model
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

    # current implementation; initialize with constant to compare outputs
    mistral_model = mistral(
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
    fixed_init_model(mistral_model)

    with torch.no_grad():
        mistral_model_out = mistral_model(x_input)

    # initialize reference implementation with constant weights
    ref_mistral_model = Transformer(
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

    mapped_sd = {}
    for k, v in mistral_model.state_dict().items():
        new_k = k.replace("attn", "attention")
        new_k = (
            new_k.replace("q_proj", "wq")
            .replace("k_proj", "wk")
            .replace("v_proj", "wv")
            .replace("output_proj", "wo")
        )
        new_k = new_k.replace("mlp", "feed_forward")
        new_k = new_k.replace("feed_forward_norm.scale", "ffn_norm.weight")
        new_k = new_k.replace("sa_norm.scale", "attention_norm.weight")

        new_k = new_k.replace("norm.scale", "norm.weight")
        mapped_sd[new_k] = v

    ref_mistral_model.load_state_dict(mapped_sd)

    with torch.no_grad():
        red_mistral_model_out = ref_mistral_model(x_input, torch.arange(seq_len))

    # # value: torch.tensor(18.2749)
    print(f"mistral_model_out.mean(): {mistral_model_out.mean()}")
    print(f"red_mistral_model_out.mean(): {red_mistral_model_out.mean()}")

    torch.testing.assert_close(
        mistral_model_out, red_mistral_model_out, atol=1e-2, rtol=1e-2
    )


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
