# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ._generation import (
    generate_next_token_with_logits,
    generate_with_logits,
    get_causal_mask,
)

from ._types import PPOStats, Trajectory
from .collate import left_padded_collate, padded_collate_dpo
from .rewards import (
    estimate_advantages,
    get_reward_penalty_mask,
    get_rewards_ppo,
    masked_mean,
    masked_var,
    whiten,
)
from .sequence_processing import (
    logits_to_logprobs,
    truncate_sequence_at_first_stop_token,
    truncate_sequence_for_logprobs,
)

__all__ = [
    "generate_with_logits",
    "generate_next_token_with_logits",
    "truncate_sequence_at_first_stop_token",
    "get_causal_mask",
    "logits_to_logprobs",
    "truncate_sequence_for_logprobs",
    "get_reward_penalty_mask",
    "left_padded_collate",
    "padded_collate_dpo",
    "estimate_advantages",
    "get_rewards_ppo",
    "whiten",
    "masked_mean",
    "masked_var",
    "PPOStats",
    "Trajectory",
]
