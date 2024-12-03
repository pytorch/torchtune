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

def calculate_mfu_and_flops(
    model_flops: float,
    batch_size: int,
    step_time: float,
    world_size: int = 1,
    device: Optional[torch.device] = None,
) -> tuple[float, float]:
    """Calculate Model FLOPs Utilization and actual FLOPS.
    
    Args:
        model_flops: Number of FLOPs for one forward pass
        batch_size: Batch size used in training
        step_time: Time taken for one step in seconds
        world_size: Number of GPUs used in training
        device: Optional device to use for calculation. If None, uses current device.
    
    Returns:
        tuple[float, float]: A tuple containing:
            - MFU as a percentage (0-100)
            - Actual FLOPS achieved (operations per second)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if device.type != "cuda":
        return 0.0, 0.0
        
    peak_flops = get_gpu_peak_flops() * world_size
    if peak_flops == 0:
        return 0.0, 0.0
        
    actual_flops = (model_flops * batch_size) / step_time
    mfu = (actual_flops / peak_flops) * 100
    
    return mfu, actual_flops

def get_transformer_flops(
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    seq_len: int,
) -> float:
    """Calculate FLOPs for one forward pass of a transformer model.
    
    Args:
        hidden_size: Hidden size of the model
        intermediate_size: Intermediate size in MLP layers
        num_layers: Number of transformer layers
        seq_len: Sequence length
    
    Returns:
        float: Number of FLOPs for one forward pass
    """
    # FLOPs per attention layer
    qkv_flops = 3 * hidden_size * hidden_size  # QKV projections
    attn_scores_flops = seq_len * hidden_size * seq_len  # Attention scores
    attn_output_flops = seq_len * seq_len * hidden_size  # Attention output
    attn_proj_flops = hidden_size * hidden_size  # Output projection
    
    # FLOPs per MLP layer
    mlp_flops = 2 * hidden_size * intermediate_size  # Two linear layers
    
    # Total FLOPs per layer
    flops_per_layer = (
        qkv_flops + attn_scores_flops + attn_output_flops + attn_proj_flops + mlp_flops
    )
    
    # Total FLOPs for all layers
    total_flops = flops_per_layer * num_layers
    
    # Each operation above is a matrix multiplication which uses 2 FLOPs per operation
    # (one multiply and one add)
    return total_flops * 2