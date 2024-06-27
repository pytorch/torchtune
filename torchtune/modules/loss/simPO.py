# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class simPOLoss(nn.Module):
    """
    SimPO: Simple Preference Optimization with a Reference-Free Reward: https://arxiv.org/abs/2405.14734.

    Based on the implementation:
    https://github.com/princeton-nlp/SimPO/blob/main/scripts/simpo_trainer.py

    Args:
        beta (float): Temperature parameter for the DPO loss, typically in the range of 0.1 to 0.5. Default is 0.1.
        gamma (float): 
        loss_type (str): Type of loss function to be used. Should be one of ['sigmoid', 'hinge'].
    """

    def __init__(
        self,
        beta: float = 0.1,
        gamma: float = 0.5,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
    ):
        super(simPOLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing,
        self.loss_type = loss_type

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the SimPO loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the SimPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.

        Raises:
            ValueError: If an unknown loss type is specified.
        """
      
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        gamma_logratios = self.gamma / self.beta 
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - gamma_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )

        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards
