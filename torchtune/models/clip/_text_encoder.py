# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import List, Optional, Union

import torch
from torch import nn, Tensor

from torchtune.modules.attention_utils import _MaskType


class CLIPTextEncoder(nn.Module):
    """
    Text encoder for CLIP.

    Args:
        layers (Union[nn.Module, List[nn.Module], nn.ModuleList]): A single encoder layer, an
            nn.ModuleList of layers, or a list of layers.
        final_norm (nn.Module): Callable that applies normalization to the output of the encoder
        vocab_size (int): size of the vocabulary, default 49408
        max_seq_len (int): context size, default 77
        embed_dim (int): embedding/model dimension size, default 768
        num_layers (int): number of transformer layers, default 12

    Raises:
        AssertionError: num_layers is set and layer is a list
        AssertionError: num_layers is not set and layer is an nn.Module
    """

    def __init__(
        self,
        *,
        layers: Union[nn.Module, List[nn.Module], nn.ModuleList],
        final_norm: nn.Module,
        vocab_size: int = 49408,
        max_seq_len: int = 77,
        embed_dim: int = 768,
        num_layers: int = 12,
    ):
        super().__init__()
        if isinstance(layers, nn.ModuleList):
            pass
        elif isinstance(layers, list):
            layers = nn.ModuleList(layers)
        else:
            if not isinstance(layers, nn.Module):
                raise AssertionError("num_layers is defined, layers must be a module")
            if num_layers is None:
                raise AssertionError("num_layers is not defined, layers must be a list")
            layers = nn.ModuleList([copy.deepcopy(layers) for i in range(num_layers)])

        self.layers = layers
        self.final_norm = final_norm
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.empty(max_seq_len, embed_dim))

    def forward(
        self,
        tokens: Tensor,
        *,
        mask: Optional[_MaskType] = None,
    ) -> Tensor:
        """
        Args:
            tokens (Tensor): input tensor with shape ``[b x s]``
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax.
                Default is None.

        Returns:
            Tensor: output tensor with shape [b x d]

        Raises:
            ValueError: if seq_len of tokens is bigger than max_seq_len

        Shape notation:
            - b: batch size
            - s: token sequence length
            - d: token embed dim
        """
        # Input validation
        bsz, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        # Input embedding [b, s] -> [b, s, d]
        x = self.token_embedding(tokens) + self.position_embedding

        # Encoder [b, s, d] -> [b, s, d]
        for layer in self.layers:
            x = layer(
                x,
                mask=mask,
            )
        x = self.final_norm(x)

        # Select the output of the EOS token for each encoding in the batch
        # [b, s, d] -> [b, d]
        # TODO: handle the case when the EOS token is not the highest token ID
        eos_token_positions = tokens.argmax(dim=-1)
        x = x.take_along_dim(eos_token_positions.view(-1, 1, 1), dim=1).squeeze(dim=1)

        return x
