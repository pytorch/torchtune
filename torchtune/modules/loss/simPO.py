# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimPOLoss(nn.Module):
    """
    SimPO: Simple Preference Optimization with a Reference-Free Reward: https://arxiv.org/abs/2405.14734.
    Intuition from the paper:

        The effectiveness of SimPO is attributed to a key design: using the average log probability of a sequence as
        the implicit reward. Additionally, we introduce a target reward margin to the Bradley-Terry objective to
        encourage a larger margin between the winning and losing responses, further enhancing the algorithm's performance.

    Based on the TRL implementation:
    https://github.com/huggingface/trl/blob/98ad01ddfd1e1b67ec018014b83cba40e0caea66/trl/trainer/cpo_trainer.py#L603

    SimPO is pretty much identitcal to DPO but uses average logprobs to eliminate the need for a reference model to regularize
    the policy during training. It also uses a target reward margin to guide the policy towards better responses.
    This is kind of the same intuition in:class:`~torchtune.modules.loss.IPO`, but instead of optimizing against a margin
    between the reference policy and policy models, we're optimizing against a margin between the chosen and rejected responses.

    Args:
        beta (float): Equivalent temperature scaling parameter to DPO loss, typically in the range of 2.0 to 2.5. Default is 2.0.
        gamma (float): Target reward margin hyperparameter, typically we have ``gamma in (0, 1.5]``.
            Default is 0.5.
        label_smoothing (float): Parameter encoding uncertainty about the labels. Default is 0.
    """

    def __init__(
        self,
        beta: float = 2.0,
        gamma: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the SimPO loss for a batch chosen and rejected average log probabilities.

        Args:
            policy_chosen_logps (torch.Tensor): Average log probabilities of the policy model
                for the chosen responses with shape [b,].
            policy_rejected_logps (torch.Tensor): Average log probabilities of the policy model
            for the rejected responses with shape [b,].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]; A tuple of three tensors with shape [b,]:
                - losses: The SimPO loss for each example in the batch.
                - chosen_rewards: Rewards for the chosen responses.
                - rejected_rewards: Rewards for the rejected responses.
        """

        pi_logratios = policy_chosen_logps - policy_rejected_logps

        gamma_logratios = self.gamma / self.beta
        logits = pi_logratios - gamma_logratios

        losses = (
            -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )

        chosen_rewards = self.beta * (policy_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards
