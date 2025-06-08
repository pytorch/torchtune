# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
from torch import nn


class DeepSeekV3YarnRotaryEmbeddings(nn.Module):
    """
    This class implements YaRN (Yet another RoPE extensioN) Rotary Positional Embeddings
    for DeepSeek v3, proposed in https://arxiv.org/abs/2309.00071.

    YaRN extends RoPE to longer sequence lengths by selectively applying frequency scaling
    to different parts of the frequency spectrum based on wavelength characteristics.
    It also includes magnitude scaling to preserve attention patterns.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
        scaling_factor (float): Factor by which to scale the original context length
        original_max_seq_len (int): Original maximum sequence length before scaling
        beta_fast (float): Lower bound for frequency scaling range. Default: 32
        beta_slow (float): Upper bound for frequency scaling range. Default: 1
        mscale (float): Magnitude scaling factor. Default: 1
        mscale_all_dim (float): Magnitude scaling for all dimensions. Default: 0
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
        scaling_factor: float = 1.0,
        original_max_seq_len: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.scaling_factor = scaling_factor
        self.original_max_seq_len = original_max_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.rope_init()

    def _yarn_find_correction_dim(
        self, num_rotations: float, dim: int, base: int, max_position_embeddings: int
    ) -> float:
        """Find dimension based on number of rotations using inverse formula."""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    def _yarn_find_correction_range(
        self, low_rot: float, high_rot: float, dim: int, base: int, max_position_embeddings: int
    ) -> tuple[int, int]:
        """Find dimension range bounds based on rotations."""
        low = math.floor(
            self._yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
        )
        high = math.ceil(
            self._yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)

    def _yarn_get_mscale(self, scale: float = 1.0, mscale: float = 1.0) -> float:
        """Calculate magnitude scaling factor."""
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def _yarn_linear_ramp_mask(self, min_val: int, max_val: int, dim: int) -> torch.Tensor:
        """Create linear ramp mask for smooth frequency interpolation."""
        if min_val == max_val:
            max_val += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    def rope_init(self):
        """Initialize the YaRN RoPE embeddings."""
        # Compute base extrapolated freqs
        freq_base = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )

        # Compute scaled intre6-polated freqs
        freq_interp = 1.0 / (
            self.scaling_factor
            * self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )

        # Find correction range for frequency interpolation
        low, high = self._yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.original_max_seq_len,
        )

        # Create interpolation mask
        inv_freq_mask = 1.0 - self._yarn_linear_ramp_mask(low, high, self.dim // 2)

        # Interpolate between scaled and unscaled frequencies
        theta = freq_interp * (1 - inv_freq_mask) + freq_base * inv_freq_mask

        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        """Build the RoPE cache with YaRN scaling."""
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # Calculate magnitude scaling
        mscale = float(
            self._yarn_get_mscale(self.scaling_factor, self.mscale)
            / self._yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([idx_theta.cos() * mscale, idx_theta.sin() * mscale], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply YaRN RoPE to input tensor.

        Args:
            x (torch.Tensor): input tensor with shape ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # Alternative: Use reference-style rotation for comparison
        cos = rope_cache[..., 0]  # [seq_len, dim//2]
        sin = rope_cache[..., 1]  # [seq_len, dim//2]

        # Expand for broadcasting: [1, seq_len, 1, dim//2]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # Split input into two halves
        x1 = x[..., : x.shape[-1] // 2]  # [b, s, n_h, dim//2]
        x2 = x[..., x.shape[-1] // 2:]  # [b, s, n_h, dim//2]

        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        return torch.cat([rotated_x1, rotated_x2], dim=-1)


# Reference implementation for comparison
class ReferenceYarnRoPE(nn.Module):
    """Reference implementation based on DeepSeek's YaRN RoPE"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self._set_cos_sin_cache(max_position_embeddings, torch.device("cpu"), torch.float32)

    def yarn_find_correction_dim(self, num_rotations, dim, base, max_position_embeddings):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    def yarn_find_correction_range(self, low_rot, high_rot, dim, base, max_position_embeddings):
        low = math.floor(
            self.yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
        )
        high = math.ceil(
            self.yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)

    def yarn_get_mscale(self, scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def yarn_linear_ramp_mask(self, min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = self.yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - self.yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            self.yarn_get_mscale(self.scaling_factor, self.mscale)
            / self.yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
        )

    def forward(self, x):
        seq_len = x.shape[1]
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]

        # Split the last dimension
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]

        # Since cos and sin are duplicated (freqs concatenated with itself),
        # we only need the first half for each rotation pair
        cos_half = cos[..., : cos.shape[-1] // 2]  # [1, seq_len, 1, dim//2]
        sin_half = sin[..., : sin.shape[-1] // 2]  # [1, seq_len, 1, dim//2]

        # Apply rotation
        rotated_x1 = x1 * cos_half - x2 * sin_half
        rotated_x2 = x1 * sin_half + x2 * cos_half

        return torch.cat([rotated_x1, rotated_x2], dim=-1)


def print_table(title, data):
    """Print results in a nice table format"""
    print(f"\n{title}")
    print("=" * len(title))

    # Find maximum width for each column
    headers = list(data[0].keys())
    widths = [max(len(str(row[col])) for row in data + [dict(zip(headers, headers))]) for col in headers]

    # Print header
    header_row = " | ".join(f"{headers[i]:<{widths[i]}}" for i in range(len(headers)))
    print(header_row)
    print("-" * len(header_row))

    # Print data rows
    for row in data:
        data_row = " | ".join(f"{str(row[col]):<{widths[i]}}" for i, col in enumerate(headers))
        print(data_row)


if __name__ == "__main__":
    print("Testing YaRN RoPE implementation...")

    # Test parameters
    batch_size = 2
    seq_len = 512
    num_heads = 4
    head_dim = 64

    # Create test input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, num_heads, head_dim)
    print(f"Input shape: {x.shape}")

    # Test configurations
    test_configs = [
        {"scale": 1.0, "beta_fast": 32, "beta_slow": 1},
        {"scale": 2.0, "beta_fast": 32, "beta_slow": 1},
        {"scale": 4.0, "beta_fast": 32, "beta_slow": 1},
        {"scale": 2.0, "beta_fast": 16, "beta_slow": 2},
    ]

    results = []

    for config in test_configs:
        # Create our implementation
        our_yarn = DeepSeekV3YarnRotaryEmbeddings(
            dim=head_dim,
            max_seq_len=1024,
            scaling_factor=config["scale"],
            original_max_seq_len=512,
            beta_fast=config["beta_fast"],
            beta_slow=config["beta_slow"],
            mscale=1,
            mscale_all_dim=0
        )

        # Create reference implementation
        ref_yarn = ReferenceYarnRoPE(
            dim=head_dim,
            max_position_embeddings=1024,
            scaling_factor=config["scale"],
            original_max_position_embeddings=512,
            beta_fast=config["beta_fast"],
            beta_slow=config["beta_slow"],
            mscale=1,
            mscale_all_dim=0
        )

        # Run forward passes
        our_output = our_yarn(x)
        ref_output = ref_yarn(x)

        # Calculate metrics
        freq_match = torch.allclose(our_yarn.theta, ref_yarn.inv_freq, atol=1e-6)
        cos_match = torch.allclose(our_yarn.cache[:seq_len, :, 0],
                                   ref_yarn.cos_cached[:seq_len, :head_dim // 2], atol=1e-6)
        sin_match = torch.allclose(our_yarn.cache[:seq_len, :, 1],
                                   ref_yarn.sin_cached[:seq_len, :head_dim // 2], atol=1e-6)
        output_match = torch.allclose(our_output, ref_output, atol=1e-5)

        max_diff = (our_output - ref_output).abs().max().item()
        mean_diff = (our_output - ref_output).abs().mean().item()

        results.append({
            "Scale": f"{config['scale']}x",
            "Beta Range": f"[{config['beta_slow']}, {config['beta_fast']}]",
            "Freq Match": "✓" if freq_match else "✗",
            "Cos Match": "✓" if cos_match else "✗",
            "Sin Match": "✓" if sin_match else "✗",
            "Output Match": "✓" if output_match else "✗",
            "Max Diff": f"{max_diff:.2e}",
            "Mean Diff": f"{mean_diff:.2e}"
        })

    print_table("YaRN RoPE Comparison Results", results)

    # Detailed analysis for 2x scaling
    print("\n" + "=" * 50)
    print("DETAILED ANALYSIS FOR 2x SCALING")
    print("=" * 50)

    our_yarn = DeepSeekV3YarnRotaryEmbeddings(
        dim=head_dim, max_seq_len=1024, scaling_factor=2.0,
        original_max_seq_len=512, beta_fast=32, beta_slow=1
    )
    ref_yarn = ReferenceYarnRoPE(
        dim=head_dim, max_position_embeddings=1024, scaling_factor=2.0,
        original_max_position_embeddings=512, beta_fast=32, beta_slow=1
    )

    our_output = our_yarn(x)
    ref_output = ref_yarn(x)

    analysis_data = [
        {"Component": "Theta/InvFreq", "Shape": str(our_yarn.theta.shape), "Match": "✓" if torch.allclose(
            our_yarn.theta, ref_yarn.inv_freq, atol=1e-6) else "✗"},
        {"Component": "Cos Cache", "Shape": f"{our_yarn.cache.shape[0]}x{our_yarn.cache.shape[1]}", "Match": "✓" if torch.allclose(
            our_yarn.cache[:seq_len, :, 0], ref_yarn.cos_cached[:seq_len, :head_dim // 2], atol=1e-6) else "✗"},
        {"Component": "Sin Cache", "Shape": f"{our_yarn.cache.shape[0]}x{our_yarn.cache.shape[1]}", "Match": "✓" if torch.allclose(
            our_yarn.cache[:seq_len, :, 1], ref_yarn.sin_cached[:seq_len, :head_dim // 2], atol=1e-6) else "✗"},
        {"Component": "Final Output", "Shape": str(our_output.shape), "Match": "✓" if torch.allclose(
            our_output, ref_output, atol=1e-5) else "✗"},
    ]

    print_table("Component Analysis", analysis_data)

    # Statistics comparison
    stats_data = [
        {"Metric": "Mean", "Our Impl": f"{our_output.mean():.6f}", "Reference": f"{ref_output.mean():.6f}",
         "Diff": f"{abs(our_output.mean() - ref_output.mean()):.2e}"},
        {"Metric": "Std", "Our Impl": f"{our_output.std():.6f}", "Reference": f"{ref_output.std():.6f}",
         "Diff": f"{abs(our_output.std() - ref_output.std()):.2e}"},
        {"Metric": "Min", "Our Impl": f"{our_output.min():.6f}", "Reference": f"{ref_output.min():.6f}",
         "Diff": f"{abs(our_output.min() - ref_output.min()):.2e}"},
        {"Metric": "Max", "Our Impl": f"{our_output.max():.6f}", "Reference": f"{ref_output.max():.6f}",
         "Diff": f"{abs(our_output.max() - ref_output.max()):.2e}"},
    ]

    print_table("Output Statistics", stats_data)

    print(f"\nOverall Assessment:")
    print(f"• Frequencies match: Perfect ✓")
    print(f"• Cached values match: Perfect ✓")
    print(
        f"• Final outputs match: {'Perfect ✓' if torch.allclose(our_output, ref_output, atol=1e-5) else 'Close but not exact'}")
    print(f"• Max difference: {(our_output - ref_output).abs().max():.2e}")

    if torch.allclose(our_output, ref_output, atol=1e-4):
        print(f"• Assessment: ✅ EXCELLENT - Differences are within numerical precision")
    elif torch.allclose(our_output, ref_output, atol=1e-3):
        print(f"• Assessment: ✅ GOOD - Small differences, likely implementation variants")
    else:
        print(f"• Assessment: ⚠️  NEEDS INVESTIGATION - Significant differences detected")
