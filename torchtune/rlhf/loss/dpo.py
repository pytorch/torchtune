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
    Direct Preference Optimization (DPO) Loss module: https://arxiv.org/abs/2305.18290
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

    Note, that we still add lambda penalty from DPOP here as an option,
    but it will only provide some impact if it is bigger than 1e-5.

    Args:
        beta (float): Temperature parameter for the DPO loss, typically in the range of 0.1 to 0.5. Default is 0.1.
        label_smoothing (float): Parameter encoding uncertainty about the labels. Default is 0.
        lambda_dpop (float): The weight penalty factor in DPOP method. Default is 0.0.
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        lambda_dpop: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.lambda_dpop = lambda_dpop

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

        penalty = torch.maximum(
            torch.zeros(policy_chosen_logps.shape),
            reference_chosen_logps - policy_chosen_logps,
        )

        logits += -self.lambda_dpop * penalty

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


class DPOPLoss(nn.Module):
    """
    DPO-Positive (DPOP) Loss module: https://arxiv.org/abs/2402.13228
    Simply stated from the paper:

    In this work, first we show theoretically that the standard DPO loss can lead to a reduction
    of the model’s likelihood of the preferred examples, as long as the relative probability between the
    preferred and dispreferred classes increases

    DPOP is similar to DPO (https://arxiv.org/abs/2305.18290). The only difference is new lambda_dpop which helps to avoid
    failure mode.

    Args:
        beta (float): Temperature parameter for the DPO loss, typically in the range of 0.1 to 0.5. Default is 0.1.
        label_smoothing (float): Parameter encoding uncertainty about the labels. Default is 0.
        lambda_dpop (float): The weight penalty factor in DPOP method. Default is 0.0.
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        lambda_dpop: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.lambda_dpop = lambda_dpop

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Сompute the DPOP loss for a batch of policy and reference model log probabilities.

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
        positive_reg = reference_chosen_logps - policy_chosen_logps

        losses = -(
            F.logsigmoid(self.beta * logits)
            - self.lambda_dpop * torch.clamp(positive_reg, min=0)
        )

        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards


class RPOLoss(nn.Module):
    """
    Iterative Reasoning Preference Optimization(RPO) Loss module: https://arxiv.org/pdf/2404.19733
    Simply stated from the paper:

         We train using a modified DPO loss with an additional
         negative log-likelihood term, which we find to be crucial. We show reasoning
         improves across repeated iterations of this scheme

    Based on the implementation in HF's TRL library:
    https://github.com/huggingface/trl/blob/5d1deb1445828cfd0e947cb3a7925b1c03a283fc/trl/trainer/dpo_trainer.py#L844

    RPO is similar to DPO (https://arxiv.org/abs/2305.18290). The only difference is new nll loss which we add to computed
    loss with weighting coefficient rpo_alpha.

    Args:
        beta (float): Temperature parameter for the DPO loss, typically in the range of 0.1 to 0.5. Default is 0.1.
        label_smoothing (float): Parameter encoding uncertainty about the labels. Default is 0.
        rpo_alpha (float): Weighting coefficient to NLL loss. Default is 1.
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        rpo_alpha: float = 1.0,
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.rpo_alpha = rpo_alpha

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        nll_loss: torch.Tensor,
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
            nll_loss (torch.Tensor): NLL Loss calculated from chosen labels. Shape: (batch_size)

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

        losses = losses + self.rpo_alpha * nll_loss

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
    This is kind of the same intuition as in :class:`~torchtune.rlhf.loss.IPOLoss`, but instead of optimizing against
    a margin between the reference policy and policy models, we're optimizing against a margin between the chosen and
    rejected responses.

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
