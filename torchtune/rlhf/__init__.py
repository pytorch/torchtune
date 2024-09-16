# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from ._types import PPOStats, Trajectory
from .rewards import (
    estimate_advantages,
    get_reward_penalty_mask,
    get_rewards_ppo,
    masked_mean,
    masked_var,
    whiten,
)
from .sequence_processing import (
    get_batch_log_probs,
    logits_to_logprobs,
    truncate_sequence_at_first_stop_token,
    truncate_sequence_for_logprobs,
)

__all__ = [
    "truncate_sequence_at_first_stop_token",
    "logits_to_logprobs",
    "truncate_sequence_for_logprobs",
    "get_reward_penalty_mask",
    "estimate_advantages",
    "get_rewards_ppo",
    "whiten",
    "masked_mean",
    "masked_var",
    "PPOStats",
    "get_batch_log_probs",
    "Trajectory",
]
