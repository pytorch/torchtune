# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from tests.test_utils import assert_expected, mps_ignored_test
from torch import tensor

from torchtune.modules.position_embeddings import (
    RotaryPositionalEmbeddings,
    VisionRotaryPositionalEmbeddings,
)
from torchtune.training.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(0)


class TestRotaryPositionEmbedding:
    """
    Class for testing our Rotary Positional Embeddings (RoPE)
    implementation. The expected tensors are computed from the
    reference implementation here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py#L450
    """

    EXPECTED_X_OUT_MEAN = tensor(6.4543e-05)
    EXPECTED_X_OUT_SUM = tensor(2165.7053)
    EXPECTED_X_OUT_MAX = tensor(5.4546)

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
    def rope(self, input_params) -> RotaryPositionalEmbeddings:
        _, _, head_dim, _, max_seq_len = input_params
        return RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)

    @mps_ignored_test()
    def test_forward(self, input: tensor, rope: RotaryPositionalEmbeddings) -> None:
        x_out = rope(input)

        # check the numerics of the computed tensor
        assert_expected(x_out.mean(), self.EXPECTED_X_OUT_MEAN)
        assert_expected(x_out.sum(), self.EXPECTED_X_OUT_SUM)
        assert_expected(x_out.max(), self.EXPECTED_X_OUT_MAX)

        # check shapes
        assert_expected(x_out.shape, input.shape)

    @mps_ignored_test()
    def test_forward_with_curr_pos(
        self, input: tensor, rope: RotaryPositionalEmbeddings
    ) -> None:
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
    def test_forward_with_packed_pos(
        self, input: tensor, rope: RotaryPositionalEmbeddings
    ) -> None:
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
        rope_on_device = RotaryPositionalEmbeddings(
            dim=head_dim, max_seq_len=max_seq_len
        )
        with torch.device("meta"):
            meta_rope = RotaryPositionalEmbeddings(
                dim=head_dim, max_seq_len=max_seq_len
            )

        meta_rope.rope_init()
        for p1, p2 in zip(rope_on_device.buffers(), meta_rope.buffers()):
            torch.testing.assert_close(p1, p2)


class TestVisionRotaryPositionEmbedding:

    EXPECTED_X_OUT_MEAN = tensor(-0.00903320)
    EXPECTED_X_OUT_SUM = tensor(-29.48437119)
    EXPECTED_X_OUT_MAX = tensor(4.07074356)

    @pytest.fixture
    def input_params(self):
        bsz = 2
        max_num_tiles = 3
        num_heads = 8
        embed_dim = 32
        head_dim = embed_dim // num_heads
        patch_size = 4
        tile_size = 16
        seq_len = ((tile_size // patch_size) ** 2 + 1) * max_num_tiles
        return bsz, num_heads, head_dim, seq_len, max_num_tiles, patch_size, tile_size

    @pytest.fixture
    def input(self, input_params) -> tensor:
        bsz, num_heads, head_dim, seq_len, *_ = input_params
        return torch.randn(bsz, seq_len, num_heads, head_dim)

    @pytest.fixture
    def rope(self, input_params):
        _, _, head_dim, _, max_num_tiles, patch_size, tile_size = input_params
        return VisionRotaryPositionalEmbeddings(
            patch_size=patch_size,
            tile_size=tile_size,
            dim=head_dim,
        )

    @mps_ignored_test()
    def test_forward(self, input, rope) -> None:
        x_out = rope(input)

        # check the numerics of the computed tensor
        assert_expected(x_out.mean(), self.EXPECTED_X_OUT_MEAN)
        assert_expected(x_out.sum(), self.EXPECTED_X_OUT_SUM)
        assert_expected(x_out.max(), self.EXPECTED_X_OUT_MAX)

        # check shapes
        assert_expected(x_out.shape, input.shape)

    def test_rope_init_meta_device(self, input_params):
        _, _, head_dim, _, max_num_tiles, patch_size, tile_size = input_params
        rope_on_device = VisionRotaryPositionalEmbeddings(
            dim=head_dim,
            patch_size=patch_size,
            tile_size=tile_size,
        )
        with torch.device("meta"):
            meta_rope = VisionRotaryPositionalEmbeddings(
                dim=head_dim,
                patch_size=patch_size,
                tile_size=tile_size,
            )

        meta_rope.rope_init()
        for p1, p2 in zip(rope_on_device.buffers(), meta_rope.buffers()):
            torch.testing.assert_close(p1, p2)
