# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from tests.test_utils import assert_expected, mps_ignored_test
from torch import tensor
from torchtune.models.phi3 import Phi3RotaryPositionalEmbeddings

from torchtune.training.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(0)


class TestPhi3RotaryPositionalEmbeddings:
    """
    Class for testing the Phi3 models RoPE Embeddings. The expected tensors are
    computed from the reference implementation here:
    https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/modeling_phi3.py
    """

    @pytest.fixture
    def input_params(self):
        bsz = 4
        num_heads = 32
        embed_dim = 3072
        seq_len = 60
        max_seq_len = 4096
        head_dim = embed_dim // num_heads
        return bsz, num_heads, head_dim, seq_len, max_seq_len

    @pytest.fixture
    def input(self, input_params) -> tensor:
        bsz, num_heads, head_dim, seq_len, _ = input_params
        return torch.randn(bsz, seq_len, num_heads, head_dim)

    @pytest.fixture
    def rope_phi3(self, input_params) -> Phi3RotaryPositionalEmbeddings:
        _, _, head_dim, _, max_seq_len = input_params
        return Phi3RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)

    @mps_ignored_test()
    def test_forward(
        self, input: tensor, rope_phi3: Phi3RotaryPositionalEmbeddings
    ) -> None:
        x_out = rope_phi3(input)

        # check the numerics of the computed tensor
        assert_expected(x_out.mean(), tensor(-0.0005), atol=1e-4)
        assert_expected(x_out.sum(), tensor(-381.0620))

        # check shapes
        assert_expected(x_out.shape, input.shape)
