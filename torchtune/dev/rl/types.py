# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import NamedTuple, Optional

import torch


class GRPOTrajectory(NamedTuple):
    """
    Contains a collection of tensors describing a generated trajectory during GRPO training.

    Attributes:
        query_responses (torch.Tensor): (query, response) pairs with shape [B x G, P+L].
        logprobs (torch.Tensor): Log probabilities of the generated responses with shape [B x G, L].
        ref_logprobs (torch.Tensor): Log probabilities of the generated responses using the reference policy with shape [B x G, L].
        advantages (torch.Tensor): Advantage estimates for the generated responses with shape [B x G].
        masks (torch.Tensor): Attention masks for input ids-generated responses pairs with shape [B x G, P+L, P+L].
        position_ids (torch.Tensor): Position IDs for input ids-generated responses pairs with shape [B x G, P+L].
        response_padding_masks (torch.Tensor): Padding masks for the truncated and padded generated responses with shape [B x G, L].
        seq_lens (torch.Tensor): Sequence lengths of truncated generated responses.
        answers (str): List of answers for the generated responses. [B x G]
    """

    query_responses: torch.Tensor = None  # [B x G, P+L]
    logprobs: torch.Tensor = None  # [B x G, L]
    ref_logprobs: torch.Tensor = None  # [B x G, L]
    advantages: torch.Tensor = None  # [B x G]
    masks: torch.Tensor = None  # [B x G, P+L, P+L]
    position_ids: torch.Tensor = None  # [B x G, P+L]
    response_padding_masks: torch.Tensor = None  # [B x G, L]
    seq_lens: torch.Tensor = None
    answers: str = None


class GRPOStats(NamedTuple):
    """
    Contains GRPO loss statistics (metrics).

    Attributes:
        loss (torch.Tensor): The total GRPO loss.
        policy_loss (torch.Tensor): The policy function loss.
        kl_loss (torch.Tensor): The KL divergence loss.
        ratios (torch.Tensor): The ratio between the current and old policy probabilities.
        clipfrac (torch.Tensor): The fraction of ratios that were clipped.
        approx_policy_kls (torch.Tensor): Average estimated KL divergence between the policy before and after the optimization step.
        metadata (Optional[dict]): Additional metadata to be logged.
    """

    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    ratios: torch.Tensor
    clipfrac: torch.Tensor
    approx_policy_kls: torch.Tensor
    metadata: Optional[dict] = None
