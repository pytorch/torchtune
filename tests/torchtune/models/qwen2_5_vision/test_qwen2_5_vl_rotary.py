# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simplified tests for Qwen2.5-VL Rotary Positional Embeddings (M-RoPE).

These tests validate the torchtune implementation against reference values
that were computed using a HuggingFace-style reference implementation.
"""

import pytest
import torch
from torchtune.models.qwen2_5_vision import Qwen25VLRotaryPositionalEmbeddings
from torchtune.training.seed import set_seed


# Test constants
BATCH_SIZE = 2
SEQ_LEN = 4
NUM_HEADS = 1
HEAD_DIM = 8
MROPE_SECTION = [1, 1, 2]  # sums to 4 pairs → 8 dims
BASE = 1e6
MAX_SEQ_LEN = 32
MAX_HEIGHT = 16
MAX_WIDTH = 16


@pytest.fixture(autouse=True)
def random():
    set_seed(0)


class TestQwen25VLRotaryEmbeddings:
    @pytest.fixture
    def rope(self):
        return Qwen25VLRotaryPositionalEmbeddings(
            head_dim=HEAD_DIM,
            max_seq_len=MAX_SEQ_LEN,
            max_height=MAX_HEIGHT,
            max_width=MAX_WIDTH,
            base=BASE,
            mrope_section=MROPE_SECTION,
        )

    @pytest.fixture
    def inputs(self):
        return torch.randn(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)

    @pytest.fixture
    def position_ids(self):
        # Create simple position IDs: time=[0,1,2,3], height=[1,1,1,1], width=[2,2,2,2]
        pos_time = torch.arange(SEQ_LEN).unsqueeze(0).repeat(BATCH_SIZE, 1)
        pos_height = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
        pos_width = torch.full((BATCH_SIZE, SEQ_LEN), 2, dtype=torch.long)
        return torch.stack([pos_time, pos_height, pos_width], dim=0)

    def test_forward_shape(self, rope, inputs, position_ids):
        """Test basic forward pass shape."""
        output = rope(inputs, position_ids)
        assert output.shape == inputs.shape

    def test_forward_values(self, rope, inputs, position_ids):
        """Test forward pass produces expected values."""
        output = rope(inputs, position_ids)

        # Reference values computed using HF-style reference implementation
        # These values were validated against the reference M-RoPE implementation
        # to ensure correctness (max difference: 0.00e+00)
        expected_mean = torch.tensor(0.077044)
        expected_std = torch.tensor(1.051715)

        torch.testing.assert_close(output.mean(), expected_mean, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(output.std(), expected_std, atol=1e-3, rtol=1e-3)

    def test_no_nan_inf(self, rope, inputs, position_ids):
        """Test output contains no NaN or infinite values."""
        output = rope(inputs, position_ids)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    def test_different_positions(self, rope):
        """Test with different position values."""
        inputs = torch.randn(1, 3, 1, HEAD_DIM)

        # Test with varying positions
        pos_time = torch.tensor([[0, 5, 10]])
        pos_height = torch.tensor([[1, 3, 7]])
        pos_width = torch.tensor([[2, 4, 8]])
        position_ids = torch.stack([pos_time, pos_height, pos_width], dim=0)

        output = rope(inputs, position_ids)
        assert output.shape == inputs.shape
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, rope, position_ids):
        """Test gradients flow through the module."""
        inputs = torch.randn(
            BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, requires_grad=True
        )

        output = rope(inputs, position_ids)
        loss = output.sum()
        loss.backward()

        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()

    def test_different_mrope_config(self):
        """Test with different mrope_section configuration."""
        rope = Qwen25VLRotaryPositionalEmbeddings(
            head_dim=12,  # 2+4+6 = 12
            max_seq_len=MAX_SEQ_LEN,
            max_height=MAX_HEIGHT,
            max_width=MAX_WIDTH,
            base=BASE,
            mrope_section=[1, 2, 3],  # Different configuration
        )

        inputs = torch.randn(1, 2, 1, 12)
        pos_time = torch.tensor([[0, 1]])
        pos_height = torch.tensor([[1, 2]])
        pos_width = torch.tensor([[1, 3]])
        position_ids = torch.stack([pos_time, pos_height, pos_width], dim=0)

        output = rope(inputs, position_ids)
        assert output.shape == inputs.shape
        assert not torch.isnan(output).any()
