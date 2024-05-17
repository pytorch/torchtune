# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import Tensor


def pool_sequence_logits(
    tokens: Tensor, logits: Tensor, padding_token_idx: int
) -> Tensor:
    """Pool sequence logits by selecting the predicted logits for the last non-padding token
    for each sequence in the batch.
    Args:
        tokens (Tensor): input tensor with shape [b x s]
        logits (Tensor): predicted logits for input tokens with shape [b x s x n]
        padding_token_idx (int): Padding token id used in the tokenizer.
    Returns:
        Tensor: Pooled logits with shape [b x n]
    Notation used for tensor shapes:
        - b: batch size
        - s: sequence length
        - n: number of classes
    """
    batch_size = tokens.shape[0]

    # inspired by the HF implementation:
    # https://github.com/huggingface/transformers/blob/928331381ef6ce0622c0b1ac704299046b3afa21/src/transformers/models/mistral/modeling_mistral.py#L1339

    # calculate per-batch-element sequence lengths by finding EOS padding tokens
    padding_mask = tokens == padding_token_idx
    if padding_mask.any():
        sequence_lengths = (
            padding_mask.logical_not().sum(-1).to(logits.device).sub(1).clip(0)
        )
    else:
        sequence_lengths = -1

    # grab logits for the last non-padding token for each sequence in the batch
    return logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
