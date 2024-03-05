# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from tests.test_utils import fixed_init_model

from torch import nn
from torchtune.models.llama2._model_utils import scale_hidden_dim_for_mlp

from torchtune.modules import FeedForward


"""
Reference implementation of FeedForward from:
https://github.com/facebookresearch/llama/blob/main/llama/model.py#L307

Replicating code here to minimize dependencies.
"""


class FeedForwardRef(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


def compare_feed_forward(embed_dim: int, hidden_dim: int) -> None:

    # make sure we have the right seed for generating outputs
    # this should match up the seed value set in the corresponding
    # unit test
    torch.manual_seed(0)

    # generate input tensor used by both implementations
    input_t = torch.randn(1, embed_dim)

    # reference implementation; initialize with constant to compare outputs
    ff_ref = FeedForwardRef(dim=embed_dim, hidden_dim=4 * embed_dim)
    fixed_init_model(ff_ref)

    with torch.no_grad():
        ff_out_ref = ff_ref(input_t)

    hidden_dim = _scale_hidden_dim_for_mlp(embed_dim)
    ff = FeedForward(dim=embed_dim, hidden_dim=hidden_dim, linear_class=torch.nn.Linear)
    fixed_init_model(ff)

    with torch.no_grad():
        ff_out = ff(input_t)

    torch.testing.assert_close(ff_out, ff_out_ref, atol=1e-5, rtol=1e-5)
    print(ff_out.mean())
    print(ff_out.max())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare RMS Norm implementations")
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=4096,
        help="Embedding dimension used to compute the dim for RopE",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=4096,
        help="Hidden dimension in the feed forward layer",
    )

    args = parser.parse_args()

    compare_feed_forward(args.embed_dim, args.hidden_dim)
