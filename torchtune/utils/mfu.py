"""Utilities for calculating Model FLOPs Utilization (MFU)."""

import torch
from typing import Optional, Dict, Any

def get_gpu_peak_flops() -> float:
    """Get theoretical peak FLOPS for the current GPU.
    
    Returns:
        float: Peak FLOPS for the current GPU. Returns 0 if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return 0.0
    
    # Get current device properties
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    # Calculate theoretical peak FLOPS
    # FLOPS = 2 * cores * clock_rate * operations_per_cycle
    gpu_flops = (
        2  # for fused multiply-add
        * props.multi_processor_count  # SM count
        * props.max_clock_rate * 1e3  # Convert to Hz
        * props.max_threads_per_block  # Threads per SM
    )
    
    return gpu_flops

def calculate_mfu(
    model_flops: float,
    batch_size: int,
    step_time: float,
    world_size: int = 1,
    device: Optional[torch.device] = None,
) -> float:
    """Calculate Model FLOPs Utilization.
    
    Args:
        model_flops: Number of FLOPs for one forward pass
        batch_size: Batch size used in training
        step_time: Time taken for one step in seconds
        world_size: Number of GPUs used in training
        device: Optional device to use for calculation. If None, uses current device.
    
    Returns:
        float: MFU as a percentage (0-100)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if device.type != "cuda":
        return 0.0
        
    peak_flops = get_gpu_peak_flops() * world_size
    if peak_flops == 0:
        return 0.0
        
    actual_flops = (model_flops * batch_size) / step_time
    mfu = (actual_flops / peak_flops) * 100
    
    return mfu

def get_model_flops(
    model: torch.nn.Module,
    input_shape: tuple,
    device: Optional[torch.device] = None,
) -> float:
    """Calculate FLOPs for one forward pass of the model.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch_size, seq_len)
        device: Optional device to use for calculation. If None, uses current device.
    
    Returns:
        float: Number of FLOPs for one forward pass
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        raise ImportError(
            "fvcore package not found. Please install it with: pip install fvcore"
        )
        
    if device is None:
        device = next(model.parameters()).device
        
    # Create dummy input
    batch_size, seq_len = input_shape
    dummy_input = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    # Calculate FLOPs
    flops = FlopCountAnalysis(model, (dummy_input,))
    return float(flops.total())