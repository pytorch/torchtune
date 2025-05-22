import torch
import torch.nn as nn
import math
from typing import Optional

# --- Helper Functions for YaRN ---

def yarn_find_correction_dim(num_rotations: int,
                             dim: int, # Full head dimension
                             base: float = 10000.0,
                             original_max_position_embeddings: int = 2048) -> float:
    """
    Calculates the dimension index (in the full dim space) at which a certain
    number of full rotations occur.
    """
    return (dim * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

def yarn_find_correction_range(beta_fast: int, # Min number of rotations for high freqs
                               beta_slow: int, # Max number of rotations for low freqs
                               dim: int, # Full head dimension
                               base: float = 10000.0,
                               original_max_position_embeddings: int = 2048) -> tuple[int, int]:
    """
    Finds the range of dimension indices [low_idx, high_idx] (in the dim//2 frequency space)
    that correspond to the specified rotation counts. These define the ramp for YaRN's interpolation.
    """
    # These are indices in the full dimension space (0 to dim-1)
    low_idx_full_dim = math.floor(yarn_find_correction_dim(beta_fast, dim, base, original_max_position_embeddings))
    high_idx_full_dim = math.ceil(yarn_find_correction_dim(beta_slow, dim, base, original_max_position_embeddings))

    # The ramp mask is applied to dim // 2 frequencies.
    # Each frequency element corresponds to two dimensions.
    # So, we need to map these full_dim indices to the frequency_dim (dim//2) space.
    # An index 'd' in full_dim corresponds to 'd // 2' in frequency_dim.
    # However, DeepSeek's code uses these bounds directly with a mask of length dim//2.
    # This implies that 'low_idx_full_dim' and 'high_idx_full_dim' are treated as bounds
    # for the elements of the frequency vector (which has length dim//2).
    # Let's stick to that interpretation for consistency with the reference.
    
    # Clamp values to be within valid indices for an array of length dim // 2
    # (i.e., 0 to dim//2 - 1)
    dim_half = dim // 2
    low_idx_for_mask = max(low_idx_full_dim, 0) # Should be max(low_idx_full_dim // 2, 0) if strictly mapping
    high_idx_for_mask = min(high_idx_full_dim, dim_half -1) # Should be min(high_idx_full_dim // 2, dim_half -1)

    # DeepSeek's `yarn_find_correction_range` returns `max(low,0), min(high, dim-1)`
    # and then `yarn_linear_ramp_mask` takes `dim//2` as its length.
    # The `low` and `high` are used directly as bounds for the mask of length `dim//2`.
    # This means `low` and `high` are effectively indices into the `dim//2` array.
    # So, the clamping should be against `dim_half - 1`.
    
    # Re-evaluating based on deepseek_tt.py:
    # yarn_find_correction_range(self.beta_fast, self.beta_slow, dim, ...)
    # -> low, high
    # yarn_linear_ramp_mask(low, high, dim // 2)
    # This implies `low` and `high` from `yarn_find_correction_range` are directly
    # used as bounds for the mask of length `dim // 2`.
    # The `dim` passed to `yarn_find_correction_range` is the full head_dim.
    # The `dim` passed to `yarn_linear_ramp_mask` is `head_dim // 2`.
    # The `low` and `high` values from `yarn_find_correction_range` are indices
    # that can range up to `head_dim - 1`.
    # When used in `yarn_linear_ramp_mask(low, high, head_dim // 2)`, these `low` and `high`
    # are used as the `min_val` and `max_val` for a ramp over `head_dim // 2` elements.
    # This seems to imply a scaling or interpretation of `low` and `high` within the ramp function.
    # Let's assume the `yarn_linear_ramp_mask` expects `min_val` and `max_val` to be
    # meaningful indices *within the range of `num_dims_to_mask`*.
    # The `low` and `high` from `yarn_find_correction_range` in deepseek_tt are indeed
    # clamped against `dim-1` (full dim).
    # The most direct interpretation from deepseek_tt is that `low` and `high` are used as is.

    return max(low_idx_full_dim, 0), min(high_idx_full_dim, dim -1) # Return bounds in full_dim space


def yarn_linear_ramp_mask(min_val: float, # Start boundary for the ramp (can be outside 0 to num_dims_to_mask-1)
                          max_val: float, # End boundary for the ramp
                          num_dims_to_mask: int # Length of the mask, e.g., head_dim // 2
                         ) -> torch.Tensor:
    """
    Creates a linear ramp mask. The ramp is from 0 to 1.
    Values of torch.arange(num_dims_to_mask) < min_val will be 0.
    Values > max_val will be 1.
    """
    if min_val == max_val:
        max_val += 0.001  # Avoid division by zero

    # Create points for the ramp from 0 to num_dims_to_mask-1
    dim_indices = torch.arange(num_dims_to_mask, dtype=torch.float32)
    
    # Calculate the ramp
    # (current_dim_index - ramp_start_point) / (ramp_end_point - ramp_start_point)
    linear_func = (dim_indices - min_val) / (max_val - min_val)
    ramp_func = torch.clamp(linear_func, 0, 1) # Clamp values to be between 0 and 1
    return ramp_func

def yarn_get_mscale(scaling_factor: float = 1.0, mscale_hyperparam: float = 1.0) -> float:
    """Calculates the magnitude scaling factor component for YaRN."""
    if scaling_factor <= 1.0:
        return 1.0
    return 0.1 * mscale_hyperparam * math.log(scaling_factor) + 1.0

# --- RoPE Application Helpers ---
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rotary_pos_emb(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies RoPE to a tensor.
    tensor: Shape (..., seq_len, head_dim)
    cos, sin: Shape (..., seq_len, head_dim), broadcastable with tensor.
    """
    return (tensor * cos) + (_rotate_half(tensor) * sin)


# --- YarnRotaryPositionalEmbeddings Class ---
class YarnRotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 4096, # New, extended max sequence length
        base: float = 10000.0,
        original_max_position_embeddings: int = 2048,
        scaling_factor: float = 1.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be divisible by 2 for RoPE.")

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings # Target extended length
        self.base = base
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scaling_factor = scaling_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale_hyperparam = mscale
        self.mscale_all_dim_hyperparam = mscale_all_dim
        self.dtype = dtype

        self.inv_freq = self._calculate_yarn_inv_freq()
        self.m_scale_factor = self._calculate_yarn_magnitude_scale()
        
        self.register_buffer("_inv_freq_buffer", self.inv_freq.to(self.dtype), persistent=False)

        self.cos_cached: Optional[torch.Tensor] = None
        self.sin_cached: Optional[torch.Tensor] = None
        self.max_seq_len_cached: int = 0

        # Pre-cache up to the new max length if needed, or handle dynamically
        self._build_cache(self.max_position_embeddings)

    def _calculate_yarn_inv_freq(self) -> torch.Tensor:
        dim_half = self.head_dim // 2
        # Frequencies are calculated for dim_half elements

        freq_extra = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=self.dtype) / self.head_dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor * (self.base ** (torch.arange(0, self.head_dim, 2, dtype=self.dtype) / self.head_dim))
        )

        # low_bound_for_ramp and high_bound_for_ramp are indices in the full head_dim space (0 to head_dim-1)
        # These define where the ramp starts and ends relative to the original dimensions.
        low_bound_for_ramp, high_bound_for_ramp = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.head_dim, # Full head_dim
            self.base,
            self.original_max_position_embeddings,
        )
        
        # The ramp_values are for the dim_half frequencies.
        # yarn_linear_ramp_mask(min_val, max_val, num_dims_to_mask)
        # min_val and max_val here are interpreted as points along the 0..num_dims_to_mask-1 axis.
        # If low_bound_for_ramp is an index in full_dim, for the mask of length dim_half,
        # the corresponding start point for the ramp is low_bound_for_ramp / 2.
        # This detail is critical for how the ramp aligns with the dimensions.
        # DeepSeek's code: yarn_linear_ramp_mask(low, high, dim // 2)
        # This implies 'low' and 'high' (from yarn_find_correction_range on full 'dim')
        # are directly used as the min_val and max_val for a ramp over 'dim // 2' elements.
        ramp_values = yarn_linear_ramp_mask(
            low_bound_for_ramp, # Use directly as per DeepSeek's pattern
            high_bound_for_ramp,  # Use directly
            dim_half # The number of elements in the mask
        )
        
        # Interpolation based on DeepSeek's YarnRotaryEmbedding:
        # inv_freq = freq_inter * (1 - inv_freq_mask_deepseek) + freq_extra * inv_freq_mask_deepseek
        # where inv_freq_mask_deepseek = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
        # This means: inv_freq = freq_inter * ramp_values + freq_extra * (1.0 - ramp_values)
        inv_freq_yarn = freq_inter * ramp_values + freq_extra * (1.0 - ramp_values)
        return inv_freq_yarn

    def _calculate_yarn_magnitude_scale(self) -> float:
        m_scale_numerator = yarn_get_mscale(self.scaling_factor, self.mscale_hyperparam)
        
        # If mscale_all_dim_hyperparam is 0.0, yarn_get_mscale will use 0.0 for its mscale_hyperparam,
        # resulting in a factor of 1.0 if scaling_factor > 1.0.
        m_scale_denominator = yarn_get_mscale(self.scaling_factor, self.mscale_all_dim_hyperparam)

        if abs(m_scale_denominator) < 1e-8:
            m_scale_denominator = 1.0
        return m_scale_numerator / m_scale_denominator

    def _build_cache(self, seq_len: int):
        if seq_len <= self.max_seq_len_cached and self.cos_cached is not None and self.sin_cached is not None:
            return # Cache is already sufficient

        self.max_seq_len_cached = seq_len
        
        # Ensure inv_freq is on the correct device. It should be due to register_buffer.
        current_device = self._inv_freq_buffer.device
        inv_freq_to_use = self._inv_freq_buffer.to(current_device)

        t = torch.arange(seq_len, device=current_device, dtype=self.dtype)
        freqs = torch.outer(t, inv_freq_to_use) # Shape: (seq_len, head_dim // 2)

        # Create embeddings of shape (seq_len, head_dim) for cos and sin
        # Each frequency in freqs corresponds to a pair of dimensions.
        # Standard RoPE implementations often create cos/sin for head_dim//2 and then duplicate or interleave.
        # DeepSeek's implementation (and HF's) often does:
        # emb = torch.cat((freqs, freqs), dim=-1) # (seq_len, head_dim)
        # cos_cached = emb.cos()
        # This means the same frequency is applied to two consecutive dimensions, but one gets cos, other sin via _rotate_half.
        # For direct application with _apply_rotary_pos_emb, we need cos and sin to be (seq_len, head_dim)
        # where cos[..., 0:dim//2] and cos[..., dim//2:dim] are derived from freqs, and similarly for sin.
        # More precisely, for a dimension `j`, if `j` is even, `cos_part = cos(pos * inv_freq[j//2])`,
        # if `j` is odd, `cos_part = cos(pos * inv_freq[j//2])`.
        # And for sin, if `j` is even, `sin_part = sin(pos * inv_freq[j//2])`,
        # if `j` is odd, `sin_part = sin(pos * inv_freq[j//2])`.
        # This is what `torch.cat((freqs, freqs), dim=-1)` effectively prepares for `_apply_rotary_pos_emb`.

        emb = torch.cat((freqs, freqs), dim=-1) # Shape: (seq_len, head_dim)
        
        self.cos_cached = (emb.cos() * self.m_scale_factor).to(self.dtype)
        self.sin_cached = (emb.sin() * self.m_scale_factor).to(self.dtype)
        
        # Update buffers if they exist, otherwise create them
        if hasattr(self, '_cos_cached_buffer'):
            self.register_buffer("_cos_cached_buffer", self.cos_cached, persistent=False)
            self.register_buffer("_sin_cached_buffer", self.sin_cached, persistent=False)
        else: # First time
            self.register_buffer("_cos_cached_buffer", self.cos_cached, persistent=False)
            self.register_buffer("_sin_cached_buffer", self.sin_cached, persistent=False)


    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor, e.g., query (Q) or key (K).
                              Expected shape: (batch_size, num_heads, seq_len, head_dim).
            input_pos (Optional[torch.Tensor]): Positions of tokens.
                                                Shape: (batch_size, seq_len) or (seq_len,).
                                                If None, assumes positions are [0, 1, ..., seq_len-1].
        Returns:
            torch.Tensor: Rotated tensor with the same shape as x.
        """
        batch_size, num_heads, seq_len, head_dim_x = x.shape
        assert head_dim_x == self.head_dim, "Input head_dim does not match module's head_dim"

        self._build_cache(max(seq_len, self.max_seq_len_cached)) # Ensure cache is up-to-date for current seq_len

        if input_pos is None:
            # Use positions [0, 1, ..., seq_len-1]
            # Slice the cache: (seq_len, head_dim)
            cos = self._cos_cached_buffer[:seq_len]
            sin = self._sin_cached_buffer[:seq_len]
            # Reshape for broadcasting: (1, 1, seq_len, head_dim)
            cos = cos.view(1, 1, seq_len, self.head_dim)
            sin = sin.view(1, 1, seq_len, self.head_dim)
        else:
            # input_pos shape: (batch_size, seq_len) or (seq_len for KV cache current token)
            # Gather from cache: results in (batch_size, seq_len, head_dim) or (seq_len, head_dim)
            cos = self._cos_cached_buffer[input_pos]
            sin = self._sin_cached_buffer[input_pos]
            # Reshape for broadcasting with x (bs, num_h, slen, hdim)
            if cos.ndim == 2: # (slen, hdim) - typical for KV cache current token
                cos = cos.view(1, 1, -1, self.head_dim) # (1, 1, slen_kv, hdim)
                sin = sin.view(1, 1, -1, self.head_dim) # (1, 1, slen_kv, hdim)
            elif cos.ndim == 3: # (bs, slen, hdim)
                cos = cos.unsqueeze(1) # (bs, 1, slen, hdim)
                sin = sin.unsqueeze(1) # (bs, 1, slen, hdim)
            # If input_pos was (1, slen) for a single sample in batch with full sequence
            elif cos.ndim == 2 and input_pos.ndim == 2 and input_pos.shape[0] == 1: # (1, slen, hdim)
                 cos = cos.unsqueeze(0).unsqueeze(1) # (1,1,slen,hdim)
                 sin = sin.unsqueeze(0).unsqueeze(1)


        rotated_x = _apply_rotary_pos_emb(x, cos, sin)
        return rotated_x

if __name__ == '__main__':
    # Example Usage
    HEAD_DIM = 64
    MAX_EXTENDED_LEN = 1024
    ORIGINAL_MAX_LEN = 256
    SCALING_FACTOR = MAX_EXTENDED_LEN / ORIGINAL_MAX_LEN # s = 4.0

    yarn_rope = YarnRotaryPositionalEmbeddings(
        head_dim=HEAD_DIM,
        max_position_embeddings=MAX_EXTENDED_LEN,
        base=10000.0,
        original_max_position_embeddings=ORIGINAL_MAX_LEN,
        scaling_factor=SCALING_FACTOR,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
        mscale_all_dim=0.0, # Common setting from DeepSeek config
        dtype=torch.float32
    )

    BATCH_SIZE = 2
    NUM_HEADS = 4
    SEQ_LEN_TEST = 512

    # Dummy Q tensor: (bs, num_heads, seq_len, head_dim)
    q_tensor = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_TEST, HEAD_DIM)
    
    # 1. Test with implicit positions [0, ..., SEQ_LEN_TEST-1]
    q_rotated_implicit_pos = yarn_rope(q_tensor)
    print(f"Shape of Q after YaRN RoPE (implicit positions): {q_rotated_implicit_pos.shape}")

    # 2. Test with explicit positions (e.g., for packed sequences or KV cache)
    # Example: first sample uses pos 0-511, second sample uses pos 100-611
    pos_ids_sample1 = torch.arange(SEQ_LEN_TEST)
    pos_ids_sample2 = torch.arange(100, 100 + SEQ_LEN_TEST)
    explicit_pos = torch.stack([pos_ids_sample1, pos_ids_sample2], dim=0) # (bs, seq_len)
    
    q_rotated_explicit_pos = yarn_rope(q_tensor, input_pos=explicit_pos)
    print(f"Shape of Q after YaRN RoPE (explicit positions): {q_rotated_explicit_pos.shape}")

    # 3. Test KV cache scenario (single new token position)
    # Assume current token is at position 512 (0-indexed)
    current_token_pos = torch.tensor([SEQ_LEN_TEST], dtype=torch.long) # Shape (1,) or (bs, 1)
    # For a single token, seq_len in Q/K would be 1
    k_tensor_current = torch.randn(BATCH_SIZE, NUM_HEADS, 1, HEAD_DIM)
    k_rotated_kv_cache = yarn_rope(k_tensor_current, input_pos=current_token_pos.unsqueeze(0).expand(BATCH_SIZE, -1)) # (bs, 1)
    print(f"Shape of K for current token after YaRN RoPE (KV cache): {k_rotated_kv_cache.shape}")

    # Test if cache rebuilds for longer sequence
    SEQ_LEN_LONGER = MAX_EXTENDED_LEN + 100
    q_tensor_longer = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN_LONGER, HEAD_DIM)
    print(f"Max cached before longer: {yarn_rope.max_seq_len_cached}")
    q_rotated_longer = yarn_rope(q_tensor_longer)
    print(f"Max cached after longer: {yarn_rope.max_seq_len_cached}")
    print(f"Shape of Q after YaRN RoPE (longer sequence, cache rebuild): {q_rotated_longer.shape}")