# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from tests.test_utils import assert_expected, mps_ignored_test
from torch import tensor

from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE

from torchtune.training.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(0)


class TestLlama3ScaledRoPE:
    """
    Class for testing our Scaled RoPE for LLama3.1 (RoPE)
    implementation. The expected tensors are computed from the
    reference implementation here:
    https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py#L272

    The expected values are computed using the following code:
    https://gist.github.com/joecummings/4f1331a9c1e5aa15bad1641acb74fe0e
    """

    EXPECTED_FREQS_CIS_MEAN = tensor(0.1738)
    EXPECTED_FREQS_CIS_SUM = tensor(91141.7656)
    EXPECTED_FREQS_CIS_MAX = tensor(1.0)

    EXPECTED_X_OUT_MEAN = tensor(-2.4781e-06)
    EXPECTED_X_OUT_SUM = tensor(-83.1523)
    EXPECTED_X_OUT_MAX = tensor(5.4625)

    @pytest.fixture
    def input_params(self):
        bsz = 4
        num_heads = 32
        embed_dim = 4096
        head_dim = embed_dim // num_heads
        seq_len = 2048
        max_seq_len = 4096
        return bsz, num_heads, head_dim, seq_len, max_seq_len

    @pytest.fixture
    def input(self, input_params) -> tensor:
        bsz, num_heads, head_dim, seq_len, _ = input_params
        return torch.randn(bsz, seq_len, num_heads, head_dim)

    @pytest.fixture
    def rope(self, input_params) -> Llama3ScaledRoPE:
        _, _, head_dim, _, max_seq_len = input_params
        return Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len)

    def test_cache_equality(self, input, rope) -> None:
        # Have to explicitly call _rope_init() to initialize theta matrix
        rope.rope_init()
        cache = rope.cache

        assert_expected(cache.mean(), self.EXPECTED_FREQS_CIS_MEAN, atol=1e-4)
        assert_expected(cache.sum(), self.EXPECTED_FREQS_CIS_SUM, atol=1e-4)
        assert_expected(cache.max(), self.EXPECTED_FREQS_CIS_MAX)

    @mps_ignored_test()
    def test_forward(self, input, rope) -> None:
        x_out = rope(input)

        # check the numerics of the computed tensor
        assert_expected(x_out.mean(), self.EXPECTED_X_OUT_MEAN)
        assert_expected(x_out.sum(), self.EXPECTED_X_OUT_SUM)
        assert_expected(x_out.max(), self.EXPECTED_X_OUT_MAX)

        # check shapes
        assert_expected(x_out.shape, input.shape)

    @mps_ignored_test()
    def test_forward_with_curr_pos(self, input, rope) -> None:
        (
            _,
            seq_len,
            _,
            _,
        ) = input.shape
        x_out = rope(input, input_pos=torch.arange(seq_len))

        # these values should be exactly the same as test_forward
        # since in this case input_pos covers the entire input
        # sequence. This tests that input_pos works as expected i.e.
        # extracts the embeddings for the relevant positions
        assert_expected(x_out.mean(), self.EXPECTED_X_OUT_MEAN, atol=1e-4)
        assert_expected(x_out.sum(), self.EXPECTED_X_OUT_SUM)
        assert_expected(x_out.max(), self.EXPECTED_X_OUT_MAX)

        # check shapes
        assert_expected(x_out.shape, input.shape)

    @mps_ignored_test()
    def test_forward_with_2d_pos_ids(self, input, rope) -> None:
        """
        Use input_pos to indicate positions of each token relative to its sequence
        when sample is packed.
        """
        (
            bsz,
            seq_len,
            _,
            _,
        ) = input.shape
        x_out = rope(
            input, input_pos=torch.arange(seq_len).unsqueeze(0).expand(bsz, seq_len)
        )

        # these values should be exactly the same as test_forward
        # AND test_forward_with_current_pos. In this case input_pos
        # covers the entire batch dim and is defined for each sample separately.
        # This tests that input_pos works as expected i.e.
        # extracts the embeddings for the relevant positions for each sample
        assert_expected(x_out.mean(), self.EXPECTED_X_OUT_MEAN, atol=1e-4)
        assert_expected(x_out.sum(), self.EXPECTED_X_OUT_SUM)
        assert_expected(x_out.max(), self.EXPECTED_X_OUT_MAX)

        # check shapes
        assert_expected(x_out.shape, input.shape)

    def test_rope_init_meta_device(self, input_params):
        _, _, head_dim, _, max_seq_len = input_params
        rope_on_device = Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len)
        with torch.device("meta"):
            meta_rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len)

        meta_rope.rope_init()
        for p1, p2 in zip(rope_on_device.buffers(), meta_rope.buffers()):
            torch.testing.assert_close(p1, p2)

        # Assert meta_rope cache is no longer on meta device
        assert meta_rope.cache.device != torch.device("meta")
