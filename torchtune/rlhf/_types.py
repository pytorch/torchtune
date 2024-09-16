# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple

import torch


class Trajectory(NamedTuple):
    """
    Contains a collection of tensors describing a generated trajectory during RLHF

    Attributes:
        query_responses (torch.Tensor): (query, response) pairs
            shape [b, context_length + max_generated_tokens]
        logprobs (torch.Tensor): log probabilities of the generated responses with shape [b, max_generated_tokens]
        ref_logprobs (torch.Tensor): log probabilities of the generated responses using the reference policy
            shape [b, max_generated_tokens]
        values (torch.Tensor): value estimates of the generated responses with shape [b, max_generated_tokens]
        masks (torch.Tensor): attention masks for input ids-generated responses pairs
            shape [b, context_length + max_generated_tokens, context_length + max_generated_tokens]
        position_ids (torch.Tensor): position IDs for input ids-generated responses pairs
            shape [b, context_length + max_generated_tokens]
        response_padding_masks (torch.Tensor): padding masks for the truncated and padded generated responses
            shape [b, max_generated_tokens]
        value_padding_masks (torch.Tensor): padding masks for the values with
            shape [b, max_generated_tokens]
        value_seq_idxs (torch.Tensor): indexes of the token
            after the last valid (non-padding) token in the responses with shape [b]
        scores (torch.Tensor): scores from the reward model with shape [b]
        seq_lens (torch.Tensor): sequence lengths of truncated generated responses with shape [b]
    """

    query_responses: torch.Tensor
    logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    values: torch.Tensor
    masks: torch.Tensor
    position_ids: torch.Tensor
    response_padding_masks: torch.Tensor
    value_padding_masks: torch.Tensor
    value_seq_idxs: torch.Tensor
    scores: torch.Tensor
    seq_lens: torch.Tensor


class PPOStats(NamedTuple):
    """
    Contains PPO loss statistics (metrics)

    Attributes:
        loss (torch.Tensor): The total PPO loss.
        policy_loss (torch.Tensor): The policy function loss.
        value_loss (torch.Tensor): The value function loss.
        ratios (torch.Tensor): The ratio between the current and old policy probabilities.
        clipfrac (torch.Tensor): The fraction of ratios that were clipped.
        approx_policy_kls (torch.Tensor): Average estimated KL divergence between the policy before and after the optimisation step.

    """

    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    ratios: torch.Tensor
    clipfrac: torch.Tensor
    approx_policy_kls: torch.Tensor
