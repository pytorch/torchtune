# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn.functional as F
from torchtune import rlhf
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX


def truncate_sequence_at_first_stop_token(
    sequences: torch.Tensor, stop_tokens: torch.Tensor, fill_value: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Truncates sequence(s) after the first stop token and pads with ``fill_value``.

    Args:
        sequences (torch.Tensor): tensor of shape [batch_size, sequence_length] or [sequence_length].
        stop_tokens (torch.Tensor): tensor containing stop tokens.
        fill_value (int): value to pad the sequence with after the first stop token, usually ``pad_id``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors with the same shape as ``sequences``:
            - padding_mask (torch.Tensor): a bool tensor where True indicates the token has been truncated.
            - sequences (torch.Tensor) a tensor of truncated and padded sequences.

    Example:
        >>> stop_token_ids = torch.tensor([2, 869])
        >>> fill_value = 0
        >>> sequences = torch.tensor(
        >>>     [
        >>>         [869, 30, 869],
        >>>         [2, 30, 869],
        >>>         [869, 30, 2],
        >>>         [50, 30, 869],
        >>>         [13, 30, 2],
        >>>         [13, 30, 5],
        >>>         [13, 2, 20],
        >>>         [13, 2, 2],
        >>>         [2, 2, 2],
        >>>     ]
        >>> )
        >>> eos_mask, truncated_sequences = rlhf.truncate_sequence_at_first_stop_token(
        >>>     sequences, stop_token_ids, fill_value
        >>> )
        >>> eos_mask
        >>> torch.tensor([
        >>>         [False, True, True],
        >>>         [False, True, True],
        >>>         [False, True, True],
        >>>         [False, False, False],
        >>>         [False, False, False],
        >>>         [False, False, False],
        >>>         [False, False, True],
        >>>         [False, False, True],
        >>>         [False, True, True],
        >>>     ]
        >>> )
        >>> truncated_sequences
        >>> torch.tensor([
        >>>         [869, 0, 0],
        >>>         [2, 0, 0],
        >>>         [869, 0, 0],
        >>>         [50, 30, 869],
        >>>         [13, 30, 2],
        >>>         [13, 30, 5],
        >>>         [13, 2, 0],
        >>>         [13, 2, 0],
        >>>         [2, 0, 0],
        >>>     ]
        >>> )
    """
    eos_mask = torch.isin(sequences, stop_tokens)
    seq_lens = torch.cumsum(eos_mask, dim=1)
    padding_mask = (seq_lens > 1) | ((seq_lens == 1) & ~eos_mask)
    sequences[padding_mask] = fill_value
    return padding_mask, sequences


def logits_to_logprobs(
    logits: torch.Tensor, sequences: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Converts logits corresponding to a generated sequence to logprobs over the generated tokens.

    Args:
        logits (torch.Tensor): The logits tensor of shape [b, response_length, vocab_size].
        sequences (torch.Tensor): The corresponding tokens of shape [b, response_length].
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
        return rlhf.masked_mean(per_token_log_probs, loss_mask, dim=-1)
    else:
        return (per_token_log_probs * loss_mask).sum(-1)


def truncate_sequence_for_logprobs(
    query_response_logits: torch.Tensor, context_length: int
) -> torch.Tensor:
    """
    Truncates logits generated over a sequence for estimating logprobs over the tokens in the sequence.
    This assumes the sequence is of the (query, response) format with length (context_length + response_length)
    Args:
        query_response_logits (torch.Tensor): The logits tensor of shape [b, context_length + response_length, vocab_size].
        context_length (int): The length of the context.

    Returns:
        torch.Tensor: The truncated logits for the response with shape [b, response_length, vocab_size]."""
    return query_response_logits[:, context_length - 1 : -1]
