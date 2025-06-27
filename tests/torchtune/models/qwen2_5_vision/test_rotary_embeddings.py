"""Test file for Qwen2.5-VL Rotary Embeddings (M-RoPE) implementation."""

import torch
from torch import nn
from torchtune.models.qwen2_5_vision import Qwen25VLRotaryPositionalEmbeddings


# --- Reference HuggingFace-style M-RoPE implementation for comparison ---

def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **rope_kwargs):
    """Compute default RoPE parameters."""
    if config is not None and rope_kwargs:
        raise ValueError("Unexpected arguments")
    if rope_kwargs:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        prf = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * prf)
    attention_factor = 1.0
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device).float() / dim)
    )
    return inv_freq, attention_factor


class HF_Rope(nn.Module):
    """Reference HuggingFace Qwen2-VL RotaryEmbedding implementation."""
    
    def __init__(self, config, device=None):
        super().__init__()
        inv_freq, attention_scaling = _compute_default_rope_parameters(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = attention_scaling

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: any tensor with dtype/device; position_ids: [3, B, L]
        inv = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        pos = position_ids[:, :, None, :].float()
        freqs = (inv @ pos).transpose(2, 3)  # → [3, B, L, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # → [3, B, L, head_dim]
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    d = x.shape[-1]
    x1, x2 = x[..., : d//2], x[..., d//2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """
    Apply multimodal rotary positional embedding to query and key tensors.
    
    This function splits cos/sin [3,B,L,D] into chunks according to mrope_section,
    picks appropriate chunks for each dimension, and applies rotary embedding.
    """
    mrope_pairs = mrope_section * 2   # e.g. [1,1,2]→[1,1,2,1,1,2]
    mrope_section = mrope_pairs
    
    # Split into chunks according to mrope_section
    cos_chunks = cos.split(mrope_section, dim=-1)
    sin_chunks = sin.split(mrope_section, dim=-1)
    
    # Pick time/height/width for each chunk
    cos_parts = [cos_chunks[i][i % 3] for i in range(len(cos_chunks))]
    sin_parts = [sin_chunks[i][i % 3] for i in range(len(sin_chunks))]
    
    cos_flat = torch.cat(cos_parts, dim=-1).unsqueeze(unsqueeze_dim)
    sin_flat = torch.cat(sin_parts, dim=-1).unsqueeze(unsqueeze_dim)
    
    q_out = (q * cos_flat) + (rotate_half(q) * sin_flat)
    k_out = (k * cos_flat) + (rotate_half(k) * sin_flat)
    
    return q_out, k_out


# --- Helper class for testing ---

class DummyConfig:
    """Dummy configuration class for testing."""
    def __init__(self, rope_theta=1e6, hidden_size=128, num_attention_heads=1, 
                 max_position_embeddings=100, mrope_section=None):
        self.rope_theta = rope_theta
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = {"rope_type": "default", "mrope_section": mrope_section or [1, 1, 2]}


# --- Test functions ---

def test_mrope_basic_functionality():
    """Test basic M-RoPE functionality."""
    print("Testing basic M-RoPE functionality...")
    
    try:
        # Setup
        B, L, heads, D = 2, 5, 1, 8
        mrope_section = [1, 1, 2]  # sums to 4 pairs → 8 dims
        base = 1e6
        max_seq_len = 100
        max_height = 1024
        max_width = 1024

        # Create our implementation
        our_rope = Qwen25VLRotaryPositionalEmbeddings(
            head_dim=D, 
            max_seq_len=max_seq_len, 
            max_height=max_height, 
            max_width=max_width, 
            base=base, 
            mrope_section=mrope_section, 
        )

        # Create test input
        x = torch.randn(B, L, heads, D)  # [b, s_x, num_heads, head_dim]
        
        # Create position IDs
        pos_time = torch.arange(L).unsqueeze(0).repeat(B, 1)
        pos_height = torch.full((B, L), 2)
        pos_width = torch.full((B, L), 3)
        position_ids = torch.stack([pos_time, pos_height, pos_width], dim=0)

        # Forward pass
        output = our_rope(x, position_ids)

        # Check output properties
        assert isinstance(output, torch.Tensor), "Output should be a tensor"
        assert output.shape == x.shape, f"Output shape {output.shape} should match input shape {x.shape}"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        assert torch.isfinite(output).all(), "Output should contain only finite values"

        print("✅ Basic M-RoPE functionality test passed!")
        print(f"   - Input shape: {x.shape}")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Position IDs shape: {position_ids.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic M-RoPE functionality test failed: {e}")
        return False


def test_mrope_vs_reference():
    """Test our M-RoPE implementation against reference HuggingFace implementation."""
    print("Testing M-RoPE against reference implementation...")
    
    try:
        torch.manual_seed(0)
        B, L, heads, D = 2, 5, 1, 8
        mrope_section = [1, 1, 2]  # sums to 4 pairs → 8 dims
        base = 1e6
        max_seq_len = 100
        max_height = 1024
        max_width = 1024

        # Create reference HF implementation
        cfg = DummyConfig(
            rope_theta=base,
            hidden_size=D * heads,
            num_attention_heads=heads,
            max_position_embeddings=max_seq_len,
            mrope_section=mrope_section
        )
        hf_rope = HF_Rope(cfg)

        # Create our implementation
        our_rope = Qwen25VLRotaryPositionalEmbeddings(
            head_dim=D, 
            max_seq_len=max_seq_len, 
            max_height=max_height, 
            max_width=max_width, 
            base=base, 
            mrope_section=mrope_section, 
        )

        # Create test input
        x = torch.randn(B, L, heads, D)  # [b, s_x, num_heads, head_dim]
        
        # Create position IDs: time: [0…L-1], height: all 2, width: all 3
        pos_time = torch.arange(L).unsqueeze(0).repeat(B, 1)
        pos_height = torch.full((B, L), 2)
        pos_width = torch.full((B, L), 3)
        position_ids = torch.stack([pos_time, pos_height, pos_width], dim=0)

        # Reference HF computation
        x_hf = x.transpose(1, 2)  # [b, s_x, num_heads, head_dim] -> [b, num_heads, s_x, head_dim]
        cos3, sin3 = hf_rope(x_hf, position_ids)
        q_hf, _ = apply_multimodal_rotary_pos_emb(x_hf, x_hf, cos3, sin3, mrope_section)

        # Our computation
        q_ours = our_rope(x, position_ids)
        q_ours_transposed = q_ours.transpose(1, 2)  # Match HF format for comparison

        # Compare results
        assert torch.allclose(q_hf, q_ours_transposed, atol=1e-6), "Results should match reference implementation"

        print("✅ M-RoPE vs reference test passed!")
        print(f"   - Max difference: {torch.max(torch.abs(q_hf - q_ours_transposed)).item():.2e}")
        
        return True
        
    except Exception as e:
        print(f"❌ M-RoPE vs reference test failed: {e}")
        return False


def test_mrope_different_sections():
    """Test M-RoPE with different mrope_section configurations."""
    print("Testing M-RoPE with different mrope_section configurations...")
    
    try:
        B, L, heads = 2, 4, 1
        base = 1e6
        max_seq_len = 100
        max_height = 1024
        max_width = 1024
        
        # Test different mrope_section configurations
        test_configs = [
            ([16, 24, 24], 128),  # Large head dim
            ([2, 4, 2], 16),      # Small head dim
            ([1, 1, 1], 6),       # Minimal sections
            ([4, 8, 4], 32),      # Medium sections
        ]
        
        for mrope_section, head_dim in test_configs:
            print(f"   Testing mrope_section={mrope_section}, head_dim={head_dim}")
            
            # Create implementation
            rope = Qwen25VLRotaryPositionalEmbeddings(
                head_dim=head_dim, 
                max_seq_len=max_seq_len, 
                max_height=max_height, 
                max_width=max_width, 
                base=base, 
                mrope_section=mrope_section, 
            )
            
            # Create test input
            x = torch.randn(B, L, heads, head_dim)
            
            # Create position IDs
            pos_time = torch.arange(L).unsqueeze(0).repeat(B, 1)
            pos_height = torch.randint(0, 10, (B, L))
            pos_width = torch.randint(0, 10, (B, L))
            position_ids = torch.stack([pos_time, pos_height, pos_width], dim=0)
            
            # Forward pass
            output = rope(x, position_ids)
            
            # Check output
            assert output.shape == x.shape, f"Output shape mismatch for config {mrope_section}"
            assert not torch.isnan(output).any(), f"NaN values found for config {mrope_section}"
            
            print(f"     ✓ Config {mrope_section} passed")
        
        print("✅ Different mrope_section configurations test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Different mrope_section configurations test failed: {e}")
        return False


def test_mrope_cache_boundaries():
    """Test M-RoPE with cache boundary conditions."""
    print("Testing M-RoPE cache boundary conditions...")
    
    try:
        B, L, heads, D = 2, 6, 1, 8
        mrope_section = [1, 2, 1]  # sums to 4 pairs → 8 dims
        base = 1e3
        max_seq_len = 10
        max_height = 5
        max_width = 7

        # Create implementation
        rope = Qwen25VLRotaryPositionalEmbeddings(
            head_dim=D,
            max_seq_len=max_seq_len,
            max_height=max_height,
            max_width=max_width,
            base=base,
            mrope_section=mrope_section,
        )

        # Create input
        x = torch.randn(B, L, heads, D)

        # Create position IDs that test cache boundaries
        def create_boundary_positions(max_val):
            return torch.tensor([0, max_val//2, max_val-1]).repeat(1, L//3 + 1).flatten()[:L]

        pos_time = torch.stack([create_boundary_positions(max_seq_len) for _ in range(B)], dim=0)
        pos_height = torch.stack([create_boundary_positions(max_height) for _ in range(B)], dim=0)
        pos_width = torch.stack([create_boundary_positions(max_width) for _ in range(B)], dim=0)
        position_ids = torch.stack([pos_time, pos_height, pos_width], dim=0)

        # Forward pass
        output = rope(x, position_ids)

        # Check output
        assert output.shape == x.shape, "Output shape should match input shape"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        assert torch.isfinite(output).all(), "Output should contain only finite values"

        print("✅ Cache boundary conditions test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Cache boundary conditions test failed: {e}")
        return False


def test_mrope_gradient_flow():
    """Test that gradients flow properly through M-RoPE."""
    print("Testing M-RoPE gradient flow...")
    
    try:
        B, L, heads, D = 2, 4, 1, 8
        mrope_section = [1, 1, 2]
        
        # Create implementation
        rope = Qwen25VLRotaryPositionalEmbeddings(
            head_dim=D,
            max_seq_len=100,
            max_height=100,
            max_width=100,
            base=1e6,
            mrope_section=mrope_section,
        )
        
        # Create input with gradients
        x = torch.randn(B, L, heads, D, requires_grad=True)
        
        # Create position IDs
        pos_time = torch.arange(L).unsqueeze(0).repeat(B, 1)
        pos_height = torch.full((B, L), 2)
        pos_width = torch.full((B, L), 3)
        position_ids = torch.stack([pos_time, pos_height, pos_width], dim=0)
        
        # Forward pass
        output = rope(x, position_ids)
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None, "Input should have gradients"
        assert x.grad.shape == x.shape, "Gradient shape should match input shape"
        assert not torch.isnan(x.grad).any(), "Gradients should not contain NaN values"
        
        print("✅ Gradient flow test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        return False


def run_all_tests():
    """Run all M-RoPE tests."""
    print("=" * 50)
    print("Running Qwen2.5-VL M-RoPE Tests")
    print("=" * 50)
    
    tests = [
        test_mrope_basic_functionality,
        test_mrope_vs_reference,
        test_mrope_different_sections,
        test_mrope_cache_boundaries,
        test_mrope_gradient_flow,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print("-" * 30)
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")
        
    return passed == total


if __name__ == "__main__":
    run_all_tests() 