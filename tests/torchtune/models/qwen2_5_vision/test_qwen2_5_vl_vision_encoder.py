# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simplified tests for Qwen2.5-VL Vision Encoder using standard configuration.

These tests validate the torchtune vision encoder implementation using
fixed initialization and deterministic inputs. Reference values are extracted
from HuggingFace model with identical weights (using fixed_init_model)
to ensure correctness against ground truth.

Does require a GPU to run.
"""

import pytest
import torch
from tests.test_utils import fixed_init_model, gpu_test
from torch import nn
from torchtune.models.qwen2_5_vision import qwen2_5_vision_encoder
from torchtune.training.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(42)


def create_deterministic_input():
    """Create the same deterministic input as used in the extract script."""
    set_seed(42)

    num_patches = 256
    patch_dim = 1176

    input_tensor = torch.randn(num_patches, patch_dim)
    grid_thw = torch.tensor([[1, 16, 16]])

    return input_tensor, grid_thw


def get_vision_encoder():
    """Create vision encoder with exact same parameters as extract script."""
    vision_encoder = qwen2_5_vision_encoder(
        embed_dim=1280,
        num_layers=32,
        activation=nn.SiLU(),
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        out_hidden_size=3584,
        patch_size=14,
        spatial_merge_size=2,
        window_size=112,
        full_att_block_indexes=[7, 15, 23, 31],
        temporal_patch_size=2,
    )
    set_seed(123)
    fixed_init_model(vision_encoder, min_val=-0.02, max_val=0.02)
    return vision_encoder


@gpu_test(gpu_count=1)
def test_vision_encoder_forward():
    """Test vision encoder forward pass with fixed initialization."""
    vision_encoder = get_vision_encoder().cuda()

    image_tensor, grid_thw = create_deterministic_input()
    image_tensor = image_tensor.cuda()
    grid_thw = grid_thw.cuda()

    output = vision_encoder(image_tensor, grid_thw)

    expected_patches = 256 // (2 * 2)

    assert output.shape == (expected_patches, 3584)
    assert not torch.isnan(output).any()
    assert torch.isfinite(output).all()

    expected_mean = torch.tensor(0.005719).cuda()
    expected_std = torch.tensor(9.958812).cuda()
    expected_max_abs = torch.tensor(17.250065).cuda()

    torch.testing.assert_close(output.mean(), expected_mean, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(output.std(), expected_std, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        output.abs().max(), expected_max_abs, atol=1e-3, rtol=1e-3
    )


@gpu_test(gpu_count=1)
def test_vision_encoder_no_nan():
    """Test that vision encoder doesn't produce NaN values."""
    vision_encoder = get_vision_encoder().cuda()

    image_tensor, grid_thw = create_deterministic_input()
    image_tensor = image_tensor.cuda()
    grid_thw = grid_thw.cuda()

    output = vision_encoder(image_tensor, grid_thw)

    assert not torch.isnan(output).any()
    assert torch.isfinite(output).all()


@gpu_test(gpu_count=1)
def test_vision_encoder_deterministic():
    """Test that vision encoder produces deterministic outputs."""
    vision_encoder = get_vision_encoder().cuda()

    image_tensor, grid_thw = create_deterministic_input()
    image_tensor = image_tensor.cuda()
    grid_thw = grid_thw.cuda()

    output1 = vision_encoder(image_tensor, grid_thw)
    output2 = vision_encoder(image_tensor, grid_thw)

    torch.testing.assert_close(output1, output2)


@gpu_test(gpu_count=1)
def test_vision_encoder_different_grid_sizes():
    """Test vision encoder with different grid sizes."""
    vision_encoder = get_vision_encoder().cuda()

    test_configs = [
        (64, [1, 8, 8]),  # 8x8 grid
        (36, [1, 6, 6]),  # 6x6 grid
        (16, [1, 4, 4]),  # 4x4 grid
    ]

    for num_patches, grid_shape in test_configs:
        set_seed(42)
        image_tensor = torch.randn(num_patches, 1176).cuda()
        grid_thw = torch.tensor([grid_shape]).cuda()
        output = vision_encoder(image_tensor, grid_thw)

        expected_patches = num_patches // 4
        assert output.shape == (expected_patches, 3584)
        assert not torch.isnan(output).any()


@gpu_test(gpu_count=1)
def test_vision_encoder_gradient_flow():
    """Test that gradients flow through the vision encoder."""
    vision_encoder = get_vision_encoder().cuda()

    image_tensor, grid_thw = create_deterministic_input()
    image_tensor = image_tensor.cuda().requires_grad_(True)
    grid_thw = grid_thw.cuda()

    output = vision_encoder(image_tensor, grid_thw)
    loss = output.sum()
    loss.backward()

    assert image_tensor.grad is not None
    assert image_tensor.grad.shape == image_tensor.shape
    assert not torch.isnan(image_tensor.grad).any()
