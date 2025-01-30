# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchtune.modules.attention_utils import _MaskType
from torchtune.modules.kv_cache import KVCache

logger = logging.getLogger(__name__)


class Gemma2Attention(nn.Module):
    """
    Adapated from official Google Pytorch Implementation:
    https://github.com/google/gemma_pytorch/blob/80881c2e6e797ef1913a4a705d4b40394791cc58/gemma/model.py#L213
    to match torchtune style.
    A new attention had to be added since nn.functional.scaled_dot_product_attention does allow soft capping
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
        attn_dropout (float): dropout value passed onto the
            scaled_dot_product_attention function. This argument is ignored if the
            self.training is False. Default value is 0.0.
        sliding_window_size (Optional[int]): size of the sliding window if None no sliding window is applied
        softcapping (Optional[float]): capping value used for soft caping, if None no capping is performed
        query_pre_attn_scalar (Optional[int]): value used for pre attention normalisation, if None head_dim is used instead
    Raises:
        ValueError:
            If ``num_heads % num_kv_heads != 0``, **or**
            if ``embed_dim % num_heads != 0``, **or**
            if ``attn_dropout < 0`` or ``attn_dropout > 1``, **or**
            if ``q_norm`` is defined without k_norm or vice versa
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
        sliding_window_size: Optional[int] = None,
        softcapping: Optional[float] = 50.0,
        query_pre_attn_scalar: Optional[int] = None,
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

        # gemma related parameters
        self.sliding_window_size = sliding_window_size
        self.softcapping = softcapping
        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

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
            self.kv_cache = KVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )
            self.cache_enabled = True

    def reset_cache(self):
        """Reset the key value caches.

        Raises:
            RuntimeError: if key value caches are not already setup.
        """
        if self.kv_cache is None:
            raise RuntimeError(
                "Key value caches are not setup. Call ``setup_caches()`` first."
            )
        self.kv_cache.reset()

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b x s_x x d] for the query
            y (Optional[torch.Tensor]): second input tensor with shape [b x s_y x d], is the input
                for k and v. For self attention, x=y. Optional only with kv_cache enabled.
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

        Raises:
            NotImplementedError: If ``mask`` is provided, but mask is not an instance of ``torch.Tensor``.
            ValueError: If no ``y`` input and ``kv_cache`` is not enabled.

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
        #  until flex attention implementation exists, we do not accept block masks
        if mask is not None and (not isinstance(mask, torch.Tensor)):
            raise NotImplementedError(
                "Block masks are not implemeted yet, use packed=False."
            )

        # x has shape [b, s_x, d]
        # y has shape [b, s_y, d]
        b, s_x, _ = x.shape
        s_y = y.shape[1] if y is not None else 0

        # q has shape [b, s_x, num_heads * head_dim]
        q = self.q_proj(x)

        # number of queries per key/value
        q_per_kv = self.num_heads // self.num_kv_heads
        q = q.view(b, s_x, self.num_kv_heads * q_per_kv, self.head_dim)

        # Apply positional embeddings
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)

        # [b, n_h, s_x, h_d]
        q = q.transpose(1, 2)

        # Normalize q
        if self.q_norm is not None:
            q = self.q_norm(q)

        if y is None:
            if self.kv_cache is None or not self.cache_enabled:
                raise ValueError(
                    "Must provide y input or use kv_cache to enable streaming decoding"
                )
            k = self.kv_cache.k_cache
            v = self.kv_cache.v_cache
        else:
            # Update k and v shape, positional embeddings, and normalization

            # k has shape [b, s_y, num_kv_heads * head_dim]
            # v has shape [b, s_y, num_kv_heads * head_dim]
            k = self.k_proj(y)
            v = self.v_proj(y)

            # Apply positional embeddings
            # k: [b, s_y, n_kv, h_d]
            k = k.view(b, s_y, -1, self.head_dim)
            if self.pos_embeddings is not None:
                k = self.pos_embeddings(k, input_pos=input_pos)

            # View + expand + reshape bring num_kv_heads to num_heads for k and v
            # to match q.

            # k: [b, s_y, n_kv, 1, h_d]
            # v: [b, s_y, n_kv, 1, h_d]
            k = k.view(b, s_y, self.num_kv_heads, 1, self.head_dim)
            v = v.view(b, s_y, self.num_kv_heads, 1, self.head_dim)

            # If needed, expand the key and value tensors to have the same shape
            # as the query tensor by copying values across the relevant dim
            if self.num_heads != self.num_kv_heads:
                k = k.expand(b, s_y, self.num_kv_heads, q_per_kv, self.head_dim)
                v = v.expand(b, s_y, self.num_kv_heads, q_per_kv, self.head_dim)

            # [b, s, n_h, h_d]
            k = k.reshape(b, s_y, -1, self.head_dim)
            v = v.reshape(b, s_y, -1, self.head_dim)

            # [b, n_h, s, h_d]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Normalize k
            if self.k_norm is not None:
                k = self.k_norm(k)

            # Update key-value cache
            if self.kv_cache is not None and self.cache_enabled:
                k, v = self.kv_cache.update(k, v)

        q.mul_(self.scaling)
        output = torch.matmul(
            q, k.transpose(2, 3)
        )  # [batch_size, n_local_heads, input_len, head_dim]

        # if mask is None: default to causal mask
        if mask is None:
            mask = torch.tril(
                torch.ones(
                    size=(s_x, s_x),
                    dtype=torch.bool,
                ).to(x.device)
            )

        # update masks bias to be 0 for visible tokens and -2.3819763e38 otherwise
        # this is similar to what torch sdpa is doing:
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        if mask.dtype == torch.bool:
            mask = torch.where(mask.logical_not(), -2.3819763e38, 0)

        if self.sliding_window_size is not None:
            all_ones = torch.ones_like(mask)

            sliding_mask = torch.triu(
                all_ones, -1 * self.sliding_window_size + 1
            ) * torch.tril(all_ones, self.sliding_window_size - 1)
            mask = torch.where(sliding_mask == 1, mask, -2.3819763e38)

        if mask.dim() == 3:
            # This is the case for block masks where attention is different per sample
            # we want mask to be broadcastable with output so we aim for (bs, 1, s_x, s_y)
            mask = mask.unsqueeze(1)

        if self.softcapping is not None:
            output = output / self.softcapping
            output = torch.tanh(output)
            output = output * self.softcapping

        output = output + mask
        output = F.softmax(output.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(output, v)

        # reshape the output to be the same shape as the input
        output = output.transpose(1, 2).contiguous().view(b, s_x, -1)
        return self.output_proj(output)
