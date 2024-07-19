# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) Loss module: https://arxiv.org/abs/2305.18290.
    Simply stated from the paper:

        Intuitively, the DPO update increases the relative log probability of preferred to dispreferred responses,
        but it incorporates a dynamic, per-example importance weight that prevents
        the model degeneration that we find occurs with a naive probability ratio objective.

    Based on the implementation in HF's TRL library:
    https://github.com/huggingface/trl/blob/5d1deb1445828cfd0e947cb3a7925b1c03a283fc/trl/trainer/dpo_trainer.py#L844

    DPO retains similarities to PPO (https://arxiv.org/abs/2009.01325), where it optimizes a policy
    (language) model to align with human preferences, and regularizes the loss function using a baseline
    reference (the frozen, initial language model) to prevent over-fitting to the preference dataset.
    It differs from PPO by optimizing the policy model directly using labelled preference data, rather
    than using an additional reward model to provide feedback.
    This significantly simplifies training and reduces compute overhead.

    Args:
        beta (float): Temperature parameter for the DPO loss, typically in the range of 0.1 to 0.5. Default is 0.1.
        label_smoothing (float): Parameter encoding uncertainty about the labels. Default is 0.
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps (torch.Tensor): Log probabilities of the policy model
                for the chosen responses. Shape: (batch_size)
            policy_rejected_logps (torch.Tensor): Log probabilities of the policy model
                for the rejected responses. Shape: (batch_size)
            reference_chosen_logps (torch.Tensor): Log probabilities of the reference model
                for the chosen responses. Shape: (batch_size)
            reference_rejected_logps (torch.Tensor): Log probabilities of the reference model
                for the rejected responses. Shape: (batch_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors:
                - losses: The DPO loss for each example in the batch.
                - chosen_rewards: Rewards for the chosen responses.
                - rejected_rewards: Rewards for the rejected responses.

        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        losses = (
            -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )

        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards


class RSOLoss(nn.Module):
    """
    Statistical Rejection Sampling Optimization (RSO) or "hinge" loss module: https://arxiv.org/abs/2309.06657.
    Intuition from the paper:

        DPO is a logistic regression on human preference data, and SLiC (https://arxiv.org/abs/2305.10425) is almost
        equivalent to a support vector machine (SVM) with hinge loss. [RSO] improve[s] SLiC as the SVM counter part of DPO.

    Based on the implementation in HF's TRL library:
    https://github.com/huggingface/trl/blob/4dce042a3863db1d375358e8c8092b874b02934b/trl/trainer/dpo_trainer.py#L1141

    Args:
        gamma (float): Equivalent temperature parameter (from DPO) for the RSO loss.
    """

    def __init__(
        self,
        gamma: float = 0.1,
    ):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the RSO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps (torch.Tensor): Log probabilities of the policy model
                for the chosen responses. Shape: (batch_size)
            policy_rejected_logps (torch.Tensor): Log probabilities of the policy model
                for the rejected responses. Shape: (batch_size)
            reference_chosen_logps (torch.Tensor): Log probabilities of the reference model
                for the chosen responses. Shape: (batch_size)
            reference_rejected_logps (torch.Tensor): Log probabilities of the reference model
                for the rejected responses. Shape: (batch_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors:
                - losses: The RSO loss for each example in the batch.
                - chosen_rewards: Rewards for the chosen responses.
                - rejected_rewards: Rewards for the rejected responses.

        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        losses = torch.relu(1 - self.gamma * logits)

        chosen_rewards = (
            self.gamma * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.gamma * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards


class IPOLoss(nn.Module):
    """
    Identity Preference Optimisation (IPO) Loss module: https://arxiv.org/abs/2310.12036.
    Intuition from the paper:

        (Given a policy pi and reference policy, pi_ref)

        IPO learns from preferences dataset simply by regressing the gap between log-likelihood ratios

        log(pi(chosen)/pi(rejected)) and log(pi_ref(chosen)/pi_ref(rejected))

        to 1/(2*tau), where tau is the temperature parameter. [T]he weaker the regularisation becomes, the
        higher would be the log-likelihood ratio of chosen to rejected logprobs. In other words IPO, unlike DPO,
        always regularizes its solution towards pi_ref by controlling the gap between the log-likelihood ratios

        log(pi(chosen)/pi(rejected)) and log(pi_ref(chosen)/pi_ref(rejected))

        thus avoiding the over-fitting to the preference dataset.

    Based on the implementation in HF's TRL library:
    https://github.com/huggingface/trl/blob/4dce042a3863db1d375358e8c8092b874b02934b/trl/trainer/dpo_trainer.py#L1143


    Args:
        tau (float): Equivalent temperature scaling parameter (from DPO) for the IPO loss. From the TRL documentation:

            the [tau] parameter is the reciprocal of the gap between the log-likelihood ratios of the
            chosen vs the rejected completion pair and thus the smaller the tau the larger this gap is.
    """

    def __init__(
        self,
        tau: float = 0.1,
    ):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps (torch.Tensor): Log probabilities of the policy model
                for the chosen responses. Shape: (batch_size)
            policy_rejected_logps (torch.Tensor): Log probabilities of the policy model
                for the rejected responses. Shape: (batch_size)
            reference_chosen_logps (torch.Tensor): Log probabilities of the reference model
                for the chosen responses. Shape: (batch_size)
            reference_rejected_logps (torch.Tensor): Log probabilities of the reference model
                for the rejected responses. Shape: (batch_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors:
                - losses: The DPO loss for each example in the batch.
                - chosen_rewards: Rewards for the chosen responses.
                - rejected_rewards: Rewards for the rejected responses.

        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        losses = (logits - 1 / (2 * self.tau)) ** 2

        chosen_rewards = (
            self.tau * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.tau * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards
