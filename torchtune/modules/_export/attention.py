# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Optional

import torch
import torchtune.modules.attention as TorchTuneAttention
from torch import nn
from torchtune.modules._export.kv_cache import KVCache as InferenceKVCache
from torchtune.modules.attention_utils import _MaskType, _sdpa_or_flex_attention
from torchtune.modules.kv_cache import KVCache

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    NOTE: torch.export.export() friendly MultiHeadAttention, modified from
    torchtune.modules.attention.MultiHeadAttention
    Major differences:
    - Rewrite `if y is None` to torch.cond().
      - Logic becomes `if all values of y are NaN`, to make torch.compile() happy.
      - No input mutations in both false and true branches, so we need to copy kv
        values back into kv cache after torch.cond().
    - Added a SDPA module
      - SDPA module includes transpose and expanding kv dimensions.
      - Makes it easy to swap with custom SDPAs that are needed by the users of exported
        program.
    - Uses new kv cache
      - This potentially can be merged with torchtune.modules.kv_cache.
      - Changed += to .add_ to avoid mutating module attributes.
      - Added clone() method.

    Multi-headed attention layer with support for grouped query
    attention (GQA) introduced in https://arxiv.org/abs/2305.13245v1.

    GQA is a version of multiheaded attention (MHA) which uses fewer
    key/value heads than query heads by grouping n query heads for each
    key and value head. Multi-Query Attention is an extreme
    version where we have a single key and value head shared by all
    query heads.

    Following is an example of MHA, GQA and MQA with num_heads = 4

    (credit for the documentation:
    `litgpt.Config <https://github.com/Lightning-AI/litgpt/blob/eda1aaaf391fd689664f95487ab03dc137e213fd/litgpt/config.py>`_).


    ::

        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │         │        │                 │
        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
        ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
        │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
        └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
        ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
                MHA                    GQA                   MQA
        n_kv_heads =4          n_kv_heads=2           n_kv_heads=1

    Args:
        embed_dim (int): embedding dimension for the model
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            ``num_heads % num_kv_heads == 0``. For standard MHA set ``num_kv_heads == num_heads``,
            for GQA ``num_kv_heads < num_heads``, and for MQA set ``num_kv_heads == 1``.
        head_dim (int): dimension of each head, calculated by ``embed_dim // num_heads``.
        q_proj (nn.Module): projection layer for query.
        k_proj (nn.Module): projection layer for key.
        v_proj (nn.Module): projection layer for value.
        output_proj (nn.Module): projection layer for output.
        pos_embeddings (Optional[nn.Module]): positional embeddings layer, e.g. RotaryPositionalEmbeddings.
        q_norm (Optional[nn.Module]): normalization layer for query, e.g. RMSNorm. For decoding, this is applied
            before updating from kv_cache. This means it will only support token wide normalization and not
            batch or sequence wide normalization.
        k_norm (Optional[nn.Module]): normalization layer for key, must be set if q_norm is.
        kv_cache (Optional[KVCache]): KVCache object used to cache key and value
        max_seq_len (int): maximum sequence length supported by the model.
            This is needed to compute the RoPE Cache. Default: 4096.
        is_causal (bool): sets the default mask to causal when no mask is provided
        attn_dropout (float): dropout value passed onto the scaled_dot_product_attention function.
            Default value is 0.0.

    Raises:
        ValueError: If ``num_heads % num_kv_heads != 0``
        ValueError: If ``embed_dim % num_heads != 0``
        ValueError: If ``attn_dropout < 0`` or ``attn_dropout > 1``
        ValueError: if q_norm is defined without k_norm or vice versa
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        output_proj: nn.Module,
        pos_embeddings: Optional[nn.Module] = None,
        q_norm: Optional[nn.Module] = None,
        k_norm: Optional[nn.Module] = None,
        kv_cache: Optional[KVCache] = None,
        max_seq_len: int = 4096,
        is_causal: bool = True,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f"attn_dropout ({embed_dim}) must be between 0.0 and 1.0")

        if bool(q_norm) ^ bool(k_norm):
            raise ValueError("q and k norm must be set together")

        # Set attributes
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal

        # Set layers
        self.kv_cache = kv_cache
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.pos_embeddings = pos_embeddings

        # Use flex attention if supported and we are sample packing
        self._attention_call = _sdpa_or_flex_attention()
        self._sdpa = SDPA(
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            attn_dropout=self.attn_dropout if self.training else 0.0,
            is_causal=self.is_causal,
            attention_fn=self._attention_call,
            kv_cache=self.kv_cache,
        )

        # this flag indicates whether to update the kv-cache during forward
        # passes. when disabled, we can have the cache setup but still
        # perform normal forward passes
        self.cache_enabled = False

    def setup_cache(
        self, batch_size: int, dtype: torch.dtype, max_seq_len: int
    ) -> None:
        """Setup key value caches for attention calculation. If called
        after kv_cache is already setup, this will be skipped.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            max_seq_len (int): maximum sequence length model will be run with.
        """
        # Don't overwrite user defined kv_cache from init
        if self.kv_cache is not None:
            logger.warning(
                "Key value caches are already setup. You cannot call ``setup_caches()`` twice. Skipping."
            )
        else:
            self.kv_cache = InferenceKVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=dtype,
                transpose_cache=False,
            )
            self._sdpa.kv_cache = self.kv_cache
            self.cache_enabled = True

    def reset_cache(self):
        """Reset the key value caches."""
        if self.kv_cache is None:
            raise RuntimeError(
                "Key value caches are not setup. Call ``setup_caches()`` first."
            )
        self.kv_cache.reset()

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b x s_x x d] for the query
            y (torch.Tensor): second input tensor with shape [b x s_y x d], is the input
                for k and v. For self attention, x=y. If all values are NaN, we read from kv cache.
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with attention applied

        Notation used for tensor shapes:
            - b: batch size
            - s_x: sequence length for x
            - s_y: sequence length for y
            - n_h: num heads
            - n_kv: num kv heads
            - d: embed dim
            - h_d: head dim
        """
        # x has shape [b, s_x, d]
        # y has shape [b, s_y, d]
        b, s_x, _ = x.shape

        # q has shape [b, s_x, num_heads * head_dim]
        q = self.q_proj(x)

        # number of queries per key/value
        q_per_kv = self.num_heads // self.num_kv_heads
        q = q.view(b, s_x, self.num_kv_heads * q_per_kv, self.head_dim)

        # Apply positional embeddings
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)

        # Normalize q
        if self.q_norm is not None:
            q = self.q_norm(q)

        def calculate_kv(y):
            # Update k and v shape, positional embeddings, and normalization
            s_y = y.shape[1]
            # k has shape [b, s_y, num_kv_heads * head_dim]
            # v has shape [b, s_y, num_kv_heads * head_dim]
            k = self.k_proj(y)
            v = self.v_proj(y)

            # Apply positional embeddings
            # k: [b, s_y, n_kv, h_d]
            k = k.view(b, s_y, -1, self.head_dim)
            v = v.view(b, s_y, -1, self.head_dim)
            if self.pos_embeddings is not None:
                k = self.pos_embeddings(k, input_pos=input_pos)

            # Normalize k
            if self.k_norm is not None:
                k = self.k_norm(k)
            return k, v

        def true_fn(y):
            kv_cache = self.kv_cache.clone()
            return kv_cache.k_cache, kv_cache.v_cache, kv_cache.cache_pos

        def false_fn(y):
            k, v = calculate_kv(y)
            kv_cache = self.kv_cache.clone()
            kv_cache.update(k, v)
            return kv_cache.k_cache, kv_cache.v_cache, kv_cache.cache_pos

        # If kv cache is None, we expect y to be provided
        if self.kv_cache is None:
            assert (
                y is not None
            ), "Must provide y input or use kv_cache to enable streaming decoding"
            k, v = calculate_kv(y)
        else:
            # Expecting the k, v returning here to be the same size of self.kv_cache
            # In eager, we expect this predicate to specialize. In export, this will
            # become a SymBool so it's not specialized.
            k, v, cache_pos = torch.cond(
                torch.isnan(y).all().item(), true_fn, false_fn, (y,)
            )
            # Update key-value cache
            self.kv_cache.k_cache.copy_(k)
            self.kv_cache.v_cache.copy_(v)
            self.kv_cache.cache_pos.copy_(cache_pos)

        output = self._sdpa(q, k, v, b, s_x, mask=mask)
        return self.output_proj(output)


class SDPA(nn.Module):
    """
    TorchTune's SDPA which can be optimized and can be swapped
    out for a more efficient implementations.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_heads: int,
        head_dim: int,
        attn_dropout: float,
        is_causal: bool,
        attention_fn,
        kv_cache,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_per_kv = self.num_heads // self.num_kv_heads
        self.attn_dropout = attn_dropout
        self.is_causal = is_causal
        self._attention_fn = attention_fn
        self.kv_cache = kv_cache

    def forward(
        self,
        q: torch.Tensor,  # [b, s, n_h, h_d]
        k: torch.Tensor,  # [b, s, n_kv, h_d]
        v: torch.Tensor,  # [b, s, n_kv, h_d]
        bsz: int,
        seq_len: int,
        mask: Optional[_MaskType] = None,
    ) -> torch.Tensor:
        # View + expand + reshape bring num_kv_heads to num_heads for k and v
        # to match q.

        # [bsz, n_h, s, h_d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand the key and value tensors to have the same shape
        # as the query tensor by copying values across the relevant dim
        if self.num_heads != self.num_kv_heads:
            expand_shape = (-1, -1, self.q_per_kv, -1, -1)
            k = k.unsqueeze(2).expand(expand_shape).flatten(1, 2)
            v = v.unsqueeze(2).expand(expand_shape).flatten(1, 2)

        output = self._attention_fn(
            q,
            k,
            v,
            mask=mask,
            dropout_p=self.attn_dropout,
            is_causal=self.kv_cache is None and mask is None and self.is_causal,
        )
        # Reshape the output to be the same shape as the input
        return output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)


def _replace_mha_with_inference_mha(module: torch.nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, TorchTuneAttention.MultiHeadAttention):
            setattr(
                module,
                name,
                MultiHeadAttention(
                    embed_dim=child.embed_dim,
                    num_heads=child.num_heads,
                    num_kv_heads=child.num_kv_heads,
                    head_dim=child.head_dim,
                    q_proj=child.q_proj,
                    k_proj=child.k_proj,
                    v_proj=child.v_proj,
                    output_proj=child.output_proj,
                    pos_embeddings=child.pos_embeddings,
                    q_norm=child.q_norm,
                    k_norm=child.k_norm,
                    kv_cache=child.kv_cache,
                    max_seq_len=child.max_seq_len,
                    is_causal=child.is_causal,
                    attn_dropout=child.attn_dropout,
                ),
            )
        else:
            replace_mha_with_inference_mha(child)


def replace_mha_with_inference_mha(module: torch.nn.Module) -> torch.nn.Module:
    """
    Replace TorchTune's MHA with an inference friendly version of MHA that
    separates out the inference-related parts for further optimization.
    """
    _replace_mha_with_inference_mha(module)
    return module
