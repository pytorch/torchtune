# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ._generation import (
    generate,
    generate_next_token,
    generate_next_token_with_value_head_model,
    get_causal_mask,
    sample,
    update_stop_tokens_tracker,
)
from .collate import left_padded_collate
from .kl_controller import AdaptiveKLController, FixedKLController
from .rewards import (
    estimate_advantages,
    get_rewards,
    masked_mean,
    masked_whiten,
    whiten,
)

__all__ = [
    "generate",
    "generate_next_token",
    "generate_next_token_with_value_head_model",
    "get_causal_mask",
    "sample",
    "update_stop_tokens_tracker",
    "left_padded_collate",
    "AdaptiveKLController",
    "FixedKLController",
    "estimate_advantages",
    "get_rewards",
    "whiten",
    "masked_whiten",
    "masked_mean",
]
