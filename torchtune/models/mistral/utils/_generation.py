# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtune.models.mistral.modules import TransformerLMWithValueHead
from torchtune.utils._generation import sample


def generate_next_token_with_value_head_model(
    model: TransformerLMWithValueHead,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = None,
) -> torch.Tensor:
    """Generates the next tokens."""
    # model produces logits in [bsz, seq_length, vocab_size]
    # we want to take the last token's logits as the input to the next model call
    logits, _ = model(x, input_pos=input_pos)
    return sample(logits[:, -1], temperature, top_k)
