# test_mrope.py

import torch
from torch import nn

# --- HuggingFace-style M-RoPE implementation (minimal) ---

def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **rope_kwargs):
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
    """Minimal HuggingFace Qwen2-VL RotaryEmbedding (default rope_type)."""
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
    d = x.shape[-1]
    x1, x2 = x[..., : d//2], x[..., d//2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """
    Provided HF helper: splits cos/sin [3,B,L,D] into 6 chunks of
    real-dim sizes [pairs*2]*2, picks each chunk[i][i%3], and applies
    q,k = q·cos + rotate_half(q)·sin.
    """
    mrope_pairs = mrope_section * 2   # e.g. [1,1,2]→[1,1,2,1,1,2]
    mrope_section = mrope_pairs
    # split into six blocks
    cos_chunks = cos.split(mrope_section, dim=-1)
    sin_chunks = sin.split(mrope_section, dim=-1)
    # pick time/height/width for each block
    cos_parts = [ cos_chunks[i][i % 3] for i in range(len(cos_chunks)) ]
    sin_parts = [ sin_chunks[i][i % 3] for i in range(len(sin_chunks)) ]
    cos_flat = torch.cat(cos_parts, dim=-1).unsqueeze(unsqueeze_dim)
    sin_flat = torch.cat(sin_parts, dim=-1).unsqueeze(unsqueeze_dim)
    q_out = (q * cos_flat) + (rotate_half(q) * sin_flat)
    k_out = (k * cos_flat) + (rotate_half(k) * sin_flat)
    return q_out, k_out

# --- Our Qwen2.5-VL M-RoPE implementation ---

from torchtune.models.qwen2_5_vision import Qwen25VLRotaryPositionalEmbeddings

# --- Test cases ---

def test_mrope_identity():
    torch.manual_seed(0)
    B, heads, L, D = 2, 1, 5, 8
    mrope_section = [1, 1, 2]  # sums to 4 pairs → 8 dims
    base = 1e6
    max_seq_len = 100

    # Dummy config for HF implementation
    class DummyConfig:
        pass
    cfg = DummyConfig()
    cfg.rope_theta = base
    cfg.hidden_size = D * heads
    cfg.num_attention_heads = heads
    cfg.max_position_embeddings = max_seq_len
    cfg.rope_scaling = {"rope_type": "default", "mrope_section": mrope_section}

    # instantiate both
    hf_rope  = HF_Rope(cfg)
    our_rope = Qwen25VLRotaryPositionalEmbeddings(D, max_seq_len, base, mrope_section)

    # random input tensor and position ids
    x = torch.randn(B, heads, L, D)
    # time: [0…L-1], height: all 2, width: all 3
    pos_time   = torch.arange(L).unsqueeze(0).repeat(B, 1)
    pos_height = torch.full((B, L), 2)
    pos_width  = torch.full((B, L), 3)
    position_ids = torch.stack([pos_time, pos_height, pos_width], dim=0)

    # HF outputs
    cos3, sin3 = hf_rope(x, position_ids)
    q_hf, k_hf = apply_multimodal_rotary_pos_emb(x, x, cos3, sin3, mrope_section)

    # Our outputs
    cos_flat, sin_flat = our_rope(x, position_ids)
    q_ours = (x * cos_flat) + (rotate_half(x) * sin_flat)

    try: 
        assert torch.allclose(q_hf, q_ours, atol=1e-6)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print(f"q_hf: {q_hf[0, 0, 0, :10]}")
        print(f"q_ours: {q_ours[0, 0, 0, :10]}")
        breakpoint()
    print("✅ test_mrope_identity passed.")


def test_mrope_random():
    torch.manual_seed(42)
    B, heads, L, D = 3, 1, 7, 128
    mrope_section = [16, 24, 24]
    base = 1e6
    max_seq_len = 100

    class DummyConfig:
        pass
    cfg = DummyConfig()
    cfg.rope_theta = base
    cfg.hidden_size = D * heads
    cfg.num_attention_heads = heads
    cfg.max_position_embeddings = max_seq_len
    cfg.rope_scaling = {"rope_type": "default", "mrope_section": mrope_section}

    hf_rope  = HF_Rope(cfg)
    our_rope = Qwen25VLRotaryPositionalEmbeddings(D, max_seq_len, base, mrope_section)

    x = torch.randn(B, heads, L, D)
    # random position ids in [0, 10)
    pos_time   = torch.randint(0, 10, (B, L))
    pos_height = torch.randint(0, 10, (B, L))
    pos_width  = torch.randint(0, 10, (B, L))
    position_ids = torch.stack([pos_time, pos_height, pos_width], dim=0)

    cos3, sin3 = hf_rope(x, position_ids)
    q_hf, _    = apply_multimodal_rotary_pos_emb(x, x, cos3, sin3, mrope_section)

    q_ours = our_rope(x, position_ids)

    try: 
        assert torch.allclose(q_hf, q_ours, atol=1e-6)
    except AssertionError as e:
        print(f"AssertionError: {e}")
        print(f"q_hf: {q_hf[0, 0, 0, :10]}")
        print(f"q_ours: {q_ours[0, 0, 0, :10]}")
        breakpoint()
    print("✅ test_mrope_random passed.")


if __name__ == "__main__":
    test_mrope_identity()
    test_mrope_random()
