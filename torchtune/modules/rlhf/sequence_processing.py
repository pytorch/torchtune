# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX


def logits_to_logprobs(
    logits: torch.Tensor, sequences: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Converts logits corresponding to a generated sequence to logprobs over the generated tokens.
    Args:
        logits (torch.Tensor): The logits tensor of shape [b, response_length, vocab_size].
        sequences (torch.Tensor): The responses tensor of shape [b, response_length].
        temperature (float): The temperature to scale the logits. Default 1.0
    Returns:
        torch.Tensor: The log probabilities corresponding to each token in ``sequences``. Shape [b, response_length].
    """
    return torch.gather(
        F.log_softmax(logits / temperature, dim=-1),
        2,
        sequences.unsqueeze(-1),
    ).squeeze(-1)


def get_batch_log_probs(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id: int = CROSS_ENTROPY_IGNORE_IDX,
    return_average_logprobs: bool = False,
) -> torch.FloatTensor:
    """
    Calculate log probabilities based on provided logits and labels.

    Args:
        logits (torch.FloatTensor): direct logits output of the model of shape (b, s, v)
        labels (torch.LongTensor): ground-truth labels to compute log probs with, shape (b, s).
            Label tokens with a value of label_pad_token_id are ignored.
        label_pad_token_id (int): token id to ignore in labels.
        return_average_logprobs (bool): If True, return the average log probs across the sequence. Default
            is False. See https://github.com/eric-mitchell/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py#L96 # noqa


    Returns:
        Calculated log probs of shape (b, )

    Raises:
        ValueError: If logits and labels have different shapes.
    """

    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            "Logits (batch and sequence length dim) and labels must have the same shape."
        )

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    labels[labels == label_pad_token_id] = 0
    # take log-likelihood of the labels given our model
    per_token_log_probs = logits_to_logprobs(logits, labels, temperature=1.0)

    if return_average_logprobs:
        return (per_token_log_probs * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_log_probs * loss_mask).sum(-1)
