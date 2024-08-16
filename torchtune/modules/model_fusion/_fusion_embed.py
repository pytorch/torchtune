# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from torch import nn, Tensor


class FusionEmbedding(nn.Module):
    """Fusion embedding supports training additional special tokens while keeping
    the original embedding frozen. When fusing new models with a language model,
    there may be some additional tokens needed to support the fused language model.
    The FusionEmbedding keeps the original embeddings frozen while learning a much smaller
    second embedding for the additional tokens. During forward this module routes
    the tokens to the appropriate embedding table.

    Args:
        vocab_size (int): language model vocab size
        fusion_vocab_size (int): additional tokens for the fused model
        embed_dim (int): embedding dimension of the two embedding tables
    """

    def __init__(self, vocab_size: int, fusion_vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fusion_embedding = nn.Embedding(fusion_vocab_size, embed_dim)
        self.dim = embed_dim
        self.num_embeddings = vocab_size + fusion_vocab_size
        # TODO: Support merging the embeddings after finetuning

        # Keep FusionLayer wrappings out of the state_dict
        self._register_state_dict_hook(FusionEmbedding._state_dict_hook)
        self._register_load_state_dict_pre_hook(
            FusionEmbedding._load_state_dict_hook, with_module=True
        )
        # TODO: Switch to register_load_state_dict_pre_hook and
        # register_state_dict_pre_hook after PyTorch v2.5

    def _state_dict_hook(self, destination, prefix, keep_vars):
        """Remove "embedding" from the original embedding in the state_dict
        name. This keeps the orginal state dict name for the embedding
        from before fusing with the FusionEmbedding.

        [!Note] This update changes the order of the OrderedDict
        """
        key = "embedding.weight"
        new_key = "weight"
        destination[new_key] = destination[key]
        del destination[key]

    def _load_state_dict_hook(self, state_dict, *args, **kwargs):
        """Apply extra "embedding" prefix to the state_dict key to
        account for the FusionEmbedding wrapping.
        """
        key = "weight"
        new_key = "embedding.weight"
        state_dict[new_key] = state_dict[key]
        del state_dict[key]

    def fusion_params(self) -> List[str]:
        """
        Return fusion embedding parameters.
        """
        fusion_params = ["fusion_embedding.weight"]
        return fusion_params

    def _fused_embed(self, bs, seq_len):
        """
        Return an empty tensor the shape of the combined embedding.
        """
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        return torch.empty(bs, seq_len, self.dim, device=device, dtype=dtype)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): input integer tensor with shape
                [batch_size x seq_length]

        Returns:
            Tensor: output tensor embedding with shape
                [batch_size x seq_length x embed_dim]`

        """
        bs, seq_len = input.size()
        vocab_size = self.embedding.num_embeddings

        mask = input < vocab_size
        tokens = torch.masked_select(input, mask)
        fusion_tokens = torch.masked_select(input, ~mask) - vocab_size

        # [batch_size x num_tokens x embed_dim]
        embeds = self.embedding(tokens)
        # [batch_size x num_fusion_tokens x embed_dim]
        fusion_embeds = self.fusion_embedding(fusion_tokens)

        # [batch_size x seq_length x embed_dim]
        out = self._fused_embed(bs, seq_len)
        mask = mask.unsqueeze(-1).expand(bs, seq_len, self.dim)
        out.masked_scatter_(mask, embeds)
        out.masked_scatter_(~mask, fusion_embeds)
        return out
