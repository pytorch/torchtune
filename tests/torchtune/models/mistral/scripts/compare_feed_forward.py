# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from tests.test_utils import fixed_init_model
from tests.torchtune.models.mistral.scripts.mistral_reference import FeedForward

from tests.torchtune.models.mistral.scripts.mistral_test_config import MistralTestConfig

from torchtune.models.mistral._component_builders import mistral_mlp


def compare_feed_forward(embed_dim: int, intermediate_dim: int) -> None:

    # make sure we have the right seed for generating outputs
    # this should match up the seed value set in the corresponding
    # unit test
    torch.manual_seed(MistralTestConfig.SEED)

    # generate input tensor used by both implementations
    input_t = torch.randn(1, embed_dim)

    # reference implementation
    ff_ref = FeedForward(dim=embed_dim, hidden_dim=intermediate_dim)
    fixed_init_model(ff_ref)

    with torch.no_grad():
        ff_out_ref = ff_ref(input_t)

    ff = mistral_mlp(embed_dim, intermediate_dim)
    fixed_init_model(ff)

    with torch.no_grad():
        ff_out = ff(input_t)

    torch.testing.assert_close(ff_out, ff_out_ref, atol=1e-5, rtol=1e-5)
    print(f"ff_out.mean(): {ff_out.mean()}")
    print(f"ff_out.max(): {ff_out.max()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare FeedForward implementations")
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=MistralTestConfig.EMBED_DIM,
        help="Embedding dimension for self-attention",
    )
    parser.add_argument(
        "--intermediate_dim",
        type=int,
        default=MistralTestConfig.INTERMEDIATE_DIM,
        help="Intermediate dimension for MLP",
    )

    args = parser.parse_args()

    compare_feed_forward(args.embed_dim, args.intermediate_dim)
