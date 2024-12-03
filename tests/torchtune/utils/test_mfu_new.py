"""Tests for MFU (Model FLOPs Utilization) utilities."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from torchtune.utils import mfu


def test_get_gpu_peak_flops():
    """Test GPU peak FLOPS calculation."""
    # Mock CUDA device properties and lspci output
    mock_props = MagicMock()
    mock_props.name = "NVIDIA A100-SXM4-80GB"
    
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.current_device', return_value=0), \
         patch('torch.cuda.get_device_properties', return_value=mock_props), \
         patch('subprocess.run') as mock_run:
        # Test A100
        peak_flops = mfu.get_gpu_peak_flops()
        assert peak_flops == 312e12  # A100 peak FLOPS
        
        # Test H100 NVL
        mock_props.name = "NVIDIA H100 NVL"
        peak_flops = mfu.get_gpu_peak_flops()
        assert peak_flops == 835e12  # H100 NVL peak FLOPS
        
        # Test H100 PCIe
        mock_props.name = "NVIDIA H100 PCIe"
        peak_flops = mfu.get_gpu_peak_flops()
        assert peak_flops == 756e12  # H100 PCIe peak FLOPS
        
        # Test H100 SXM
        mock_props.name = "NVIDIA H100-SXM5"
        peak_flops = mfu.get_gpu_peak_flops()
        assert peak_flops == 989e12  # H100 SXM peak FLOPS
        
        # Test H200
        mock_props.name = "NVIDIA H200"
        peak_flops = mfu.get_gpu_peak_flops()
        assert peak_flops == 989e12  # H200 peak FLOPS
        
        # Test fallback for unknown GPU
        mock_props.name = "NVIDIA Unknown GPU"
        peak_flops = mfu.get_gpu_peak_flops()
        assert peak_flops == 312e12  # Falls back to A100


def test_get_gpu_peak_flops_no_cuda():
    """Test GPU peak FLOPS calculation when CUDA is not available."""
    with patch('torch.cuda.is_available', return_value=False):
        peak_flops = mfu.get_gpu_peak_flops()
        assert peak_flops == 0.0


def test_get_transformer_flops():
    """Test FLOPs calculation for a transformer model."""
    # Test case: Small transformer
    hidden_size = 128
    intermediate_size = 512
    num_layers = 2
    seq_len = 32
    
    flops = mfu.get_transformer_flops(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        seq_len=seq_len
    )
    
    # Expected FLOPs per layer:
    # QKV: 3 * h * h = 3 * 128 * 128
    # Attn scores: s * h * s = 32 * 128 * 32
    # Attn output: s * s * h = 32 * 32 * 128
    # Output proj: h * h = 128 * 128
    # MLP: 2 * h * i = 2 * 128 * 512
    expected_per_layer = (
        (3 * 128 * 128) +  # QKV
        (32 * 128 * 32) +  # Attn scores
        (32 * 32 * 128) +  # Attn output
        (128 * 128) +      # Output proj
        (2 * 128 * 512)    # MLP
    )
    expected_flops = expected_per_layer * num_layers * 2  # *2 for multiply-add
    
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
    """Test end-to-end MFU and FLOPS calculation with a transformer model."""
    # Test case: Small transformer
    hidden_size = 128
    intermediate_size = 512
    num_layers = 2
    seq_len = 32
    
    # Calculate expected FLOPs
    expected_per_layer = (
        (3 * hidden_size * hidden_size) +  # QKV
        (seq_len * hidden_size * seq_len) +  # Attn scores
        (seq_len * seq_len * hidden_size) +  # Attn output
        (hidden_size * hidden_size) +  # Output proj
        (2 * hidden_size * intermediate_size)  # MLP
    )
    expected_model_flops = expected_per_layer * num_layers * 2  # *2 for multiply-add
    
    # Mock GPU peak FLOPS
    with patch('torchtune.utils.mfu.get_gpu_peak_flops', return_value=1e12):  # 1 TFLOPS
        # First get model FLOPs
        model_flops = mfu.get_transformer_flops(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            seq_len=seq_len
        )
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