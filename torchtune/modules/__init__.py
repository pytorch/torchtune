# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .attention import MultiHeadAttention  # noqa
from .attention_utils import create_block_causal_mask, packed_block_causal_mask

from .classifier import classifier_model

from .common_utils import (
    delete_kv_caches,
    disable_kv_cache,
    local_kv_cache,
    reparametrize_as_dtype_state_dict_post_hook,
)
from .feed_forward import FeedForward  # noqa
from .kv_cache import KVCache  # noqa
from .layer_dropout import LayerDropout, prepare_layer_dropout  # noqa
from .layer_norm import Fp32LayerNorm  # noqa
from .low_precision import FrozenNF4Linear  # noqa
from .position_embeddings import (  # noqa
    RotaryPositionalEmbeddings,
    VisionRotaryPositionalEmbeddings,
)
from .rms_norm import rms_norm, RMSNorm  # noqa
from .tanh_gate import TanhGate  # noqa
from .tied_linear import TiedLinear  # noqa
from .transformer import (  # noqa
    TransformerCrossAttentionLayer,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)
from .vision_transformer import VisionTransformer
from .vq_embeddings import VectorQuantizedEmbeddings

__all__ = [
    "MultiHeadAttention",
    "TanhGate",
    "FeedForward",
    "FrozenNF4Linear",
    "KVCache",
    "RotaryPositionalEmbeddings",
    "VisionRotaryPositionalEmbeddings",
    "VectorQuantizedEmbeddings",
    "RMSNorm",
    "TiedLinear",
    "Fp32LayerNorm",
    "VisionTransformer",
    "TransformerDecoder",
    "TransformerSelfAttentionLayer",
    "TransformerCrossAttentionLayer",
    "reparametrize_as_dtype_state_dict_post_hook",
    "create_block_causal_mask",
    "packed_block_causal_mask",
    "local_kv_cache",
    "delete_kv_caches",
    "disable_kv_cache",
    "LayerDropout",
    "prepare_layer_dropout",
    "classifier_model",
    "rms_norm",
]
