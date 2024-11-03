# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .attention import MultiHeadAttention  # noqa
from .attention_utils import create_block_causal_mask, packed_block_causal_mask
from .common_utils import (
    delete_kv_caches,
    disable_kv_cache,
    local_kv_cache,
    reparametrize_as_dtype_state_dict_post_hook,
)
from .feed_forward import FeedForward  # noqa
from .kv_cache import KVCache  # noqa
from .layer_norm import Fp32LayerNorm  # noqa
from .low_precision import FrozenNF4Linear  # noqa
from .lr_schedulers import get_cosine_schedule_with_warmup  # noqa
from .position_embeddings import RotaryPositionalEmbeddings  # noqa
from .rms_norm import RMSNorm  # noqa
from .tanh_gate import TanhGate  # noqa
from .tied_linear import TiedLinear  # noqa
from .transformer import (  # noqa
    TiedEmbeddingTransformerDecoder,
    TransformerCrossAttentionLayer,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)
from .vision_transformer import VisionTransformer

__all__ = [
    "MultiHeadAttention",
    "TanhGate",
    "FeedForward",
    "FrozenNF4Linear",
    "KVCache",
    "RotaryPositionalEmbeddings",
    "RMSNorm",
    "TiedLinear",
    "Fp32LayerNorm",
    "VisionTransformer",
    "TransformerDecoder",
    "TiedEmbeddingTransformerDecoder",
    "TransformerSelfAttentionLayer",
    "TransformerCrossAttentionLayer",
    "reparametrize_as_dtype_state_dict_post_hook",
    "create_block_causal_mask",
    "packed_block_causal_mask",
    "local_kv_cache",
    "delete_kv_caches",
    "disable_kv_cache",
    "get_cosine_schedule_with_warmup",
]
