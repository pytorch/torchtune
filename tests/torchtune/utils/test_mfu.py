"""Tests for MFU (Model FLOPs Utilization) utilities."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from torchtune.utils import mfu


class SimpleLinear(nn.Module):
    """Simple model with known FLOPs count.
    
    For a linear layer with input size M and output size N:
    FLOPs = 2 * M * N (multiply-add for each output element)
    """
    def __init__(self, in_features=32, out_features=64):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return self.linear(x)


class SimpleMLP(nn.Module):
    """Simple MLP with known FLOPs count.
    
    For each linear layer:
    FLOPs = 2 * in_features * out_features
    Total FLOPs = sum of FLOPs for each layer
    """
    def __init__(self, in_features=32, hidden_size=64, out_features=16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)  # ReLU has negligible FLOPs
        x = self.fc2(x)
        return x


def test_get_gpu_peak_flops():
    """Test GPU peak FLOPS calculation."""
    # Mock CUDA device properties
    mock_props = MagicMock()
    mock_props.multi_processor_count = 10
    mock_props.max_clock_rate = 1000  # MHz
    mock_props.max_threads_per_block = 1024
    
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.current_device', return_value=0), \
         patch('torch.cuda.get_device_properties', return_value=mock_props):
        peak_flops = mfu.get_gpu_peak_flops()
        
        # Expected: 2 * 10 * (1000 * 1e3) * 1024
        expected_flops = 2 * 10 * (1000 * 1e3) * 1024
        assert peak_flops == expected_flops


def test_get_gpu_peak_flops_no_cuda():
    """Test GPU peak FLOPS calculation when CUDA is not available."""
    with patch('torch.cuda.is_available', return_value=False):
        peak_flops = mfu.get_gpu_peak_flops()
        assert peak_flops == 0.0


def test_get_model_flops_linear():
    """Test FLOPs calculation for a simple linear model."""
    model = SimpleLinear(in_features=32, out_features=64)
    
    # Expected FLOPs for linear layer: 2 * in_features * out_features
    expected_flops = 2 * 32 * 64
    
    flops = mfu.get_model_flops(model, input_shape=(1, 32))
    assert abs(flops - expected_flops) < 1e-5


def test_get_model_flops_mlp():
    """Test FLOPs calculation for a simple MLP model."""
    model = SimpleMLP(in_features=32, hidden_size=64, out_features=16)
    
    # Expected FLOPs:
    # First layer: 2 * 32 * 64
    # Second layer: 2 * 64 * 16
    expected_flops = (2 * 32 * 64) + (2 * 64 * 16)
    
    flops = mfu.get_model_flops(model, input_shape=(1, 32))
    assert abs(flops - expected_flops) < 1e-5


def test_calculate_mfu_and_flops():
    """Test MFU and FLOPS calculation with known values."""
    # Mock GPU peak FLOPS
    with patch('torchtune.utils.mfu.get_gpu_peak_flops', return_value=1e12):  # 1 TFLOPS
        # Test case: Model doing 0.1 TFLOPS (10% utilization)
        mfu_value, actual_flops = mfu.calculate_mfu_and_flops(
            model_flops=1e11,  # 0.1 TFLOPS
            batch_size=1,
            step_time=1.0,  # 1 second
            world_size=1,
            device=torch.device("cuda")
        )
        assert abs(mfu_value - 10.0) < 1e-5  # Should be 10%
        assert abs(actual_flops - 1e11) < 1e-5  # Should be 0.1 TFLOPS


def test_calculate_mfu_and_flops_no_cuda():
    """Test MFU and FLOPS calculation when CUDA is not available."""
    mfu_value, actual_flops = mfu.calculate_mfu_and_flops(
        model_flops=1e9,
        batch_size=1,
        step_time=1.0,
        world_size=1,
        device=torch.device("cpu")
    )
    assert mfu_value == 0.0
    assert actual_flops == 0.0


def test_calculate_mfu_and_flops_multi_gpu():
    """Test MFU and FLOPS calculation with multiple GPUs."""
    # Mock GPU peak FLOPS
    with patch('torchtune.utils.mfu.get_gpu_peak_flops', return_value=1e12):  # 1 TFLOPS per GPU
        # Test with 4 GPUs
        mfu_value, actual_flops = mfu.calculate_mfu_and_flops(
            model_flops=1e11,  # 0.1 TFLOPS
            batch_size=1,
            step_time=1.0,
            world_size=4,  # 4 GPUs
            device=torch.device("cuda")
        )
        # Total peak FLOPS = 4 TFLOPS, actual = 0.1 TFLOPS
        # MFU should be (0.1 / 4) * 100 = 2.5%
        assert abs(mfu_value - 2.5) < 1e-5
        assert abs(actual_flops - 1e11) < 1e-5  # Should still be 0.1 TFLOPS


def test_end_to_end_mfu_and_flops():
    """Test end-to-end MFU and FLOPS calculation with a real model."""
    model = SimpleMLP(in_features=32, hidden_size=64, out_features=16)
    
    # Calculate expected FLOPs
    expected_model_flops = (2 * 32 * 64) + (2 * 64 * 16)
    
    # Mock GPU peak FLOPS
    with patch('torchtune.utils.mfu.get_gpu_peak_flops', return_value=1e12):  # 1 TFLOPS
        # First get model FLOPs
        model_flops = mfu.get_model_flops(model, input_shape=(1, 32))
        assert abs(model_flops - expected_model_flops) < 1e-5
        
        # Then calculate MFU and FLOPS
        mfu_value, actual_flops = mfu.calculate_mfu_and_flops(
            model_flops=model_flops,
            batch_size=128,  # Larger batch size
            step_time=1.0,
            world_size=1,
            device=torch.device("cuda")
        )
        
        # Expected MFU = (model_flops * batch_size) / (peak_flops * step_time) * 100
        expected_mfu = (expected_model_flops * 128) / 1e12 * 100
        assert abs(mfu_value - expected_mfu) < 1e-5
        
        # Expected FLOPS = model_flops * batch_size / step_time
        expected_actual_flops = expected_model_flops * 128 / 1.0
        assert abs(actual_flops - expected_actual_flops) < 1e-5