# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch
from torch import nn


class FusionLayer(nn.Module):
    """Fusion layer as introduced in `Flamingo: a Visual Language Model for Few-Shot Learning <https://arxiv.org/abs/2204.14198>`_.

    Deep Fusion model architectures combine pretrained encoder models with pretrained
    language models by infusing the encoder outputs into the middle layers of the LLM.
    This allows the language model to interpret the enocder outputs as text and
    "understand" any modality for which you can train an encoder. To enable the language model
    to adapt to the encoder outputs, the FusionLayer fuses a new learnable layer to an existing
    decoder (language model) layer. This additional layer can take the encoder embeddings and
    learn to combine them with the token embeddings from the decoder. The module supports fusing
    the new layer before or after the original, in Flamingo the new layer is fused before the original.

    The original layer is wrapped in FusionLayer such that it maintains its original state_dict
    key and the pre-trained checkpoint isn't broken. The new layer parameters are available
    through ``fusion_params`` to separately control if they're trainable or not.

    Example:
        >>> # Original decoder style transformer
        >>> layer = nn.TransformerSelfAttentionLayer(...)
        >>> model = TransformerDecoder(layers=layer, num_layers=32, ...)
        >>>
        >>> # Fuse a cross attention layer to each self attention layer to adapt for the encoder
        >>> fusion_layer = nn.TransformerCrossAttentionLayer(...)
        >>> fused_layer = FusionLayer(layer, fusion_layer)
        >>> model = TransformerDecoder(layers=fused_layer, num_layers=32, ...)
        >>>
        >>> # Original decoder state_dict still works
        >>> model.load_state_dict(..., strict=False)

    Args:
        layer (nn.Module): original decoder layer
        fusion_layer (nn.Module): new fusion layer
        fusion_first (bool): boolean to insert fusion layer before or after the decoder layer.
    """

    def __init__(
        self, layer: nn.Module, fusion_layer: nn.Module, fusion_first: bool = True
    ):
        super().__init__()
        self.layer = layer
        self.fusion_layer = fusion_layer
        self.fusion_first = fusion_first

        # Keep FusionLayer wrappings out of the state_dict
        self._register_state_dict_hook(FusionLayer._state_dict_hook)
        self._register_load_state_dict_pre_hook(
            FusionLayer._load_state_dict_hook, with_module=True
        )
        # TODO: Switch to register_load_state_dict_pre_hook and
        # register_state_dict_pre_hook after PyTorch v2.5

    def _state_dict_hook(self, state_dict, prefix, *args, **kwargs):
        """Remove "layer" from the original layer in the state_dict
        name. This keeps the orginal state dict name for the layer
        from before fusing with the FusionLayer.

        [!Note] This update changes the order of the OrderedDict
        """
        keys = list(state_dict.keys())
        for key in keys:
            local_key = key[len(prefix) :]
            if local_key.startswith("layer"):
                new_key = prefix + local_key.replace("layer.", "")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

    def _load_state_dict_hook(self, state_dict, prefix, *args, **kwargs):
        """Apply extra "layer" prefix to the state_dict key to
        account for the FusionLayer wrapping.
        """
        keys = list(state_dict.keys())
        for key in keys:
            local_key = key[len(prefix) :]
            if not local_key.startswith("fusion_layer"):
                new_key = prefix + "layer." + local_key
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int,
        decoder_max_seq_len: int,
    ) -> None:
        """Setup key value cache for both layers.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): maximum cache sequence length for cross-attention layer.
            decoder_max_seq_len (int): maximum cache sequence length for self-attention layer.
        """
        self.layer.setup_caches(
            batch_size,
            dtype,
            encoder_max_seq_len=encoder_max_seq_len,
            decoder_max_seq_len=decoder_max_seq_len,
        )

        self.fusion_layer.setup_caches(
            batch_size,
            dtype,
            encoder_max_seq_len=encoder_max_seq_len,
            decoder_max_seq_len=decoder_max_seq_len,
        )

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup on ``self.layer``.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_setup`.
        """
        return self.layer.caches_are_setup()

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches on ``self.layer`` are enabled.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_enabled`.
        """
        return self.layer.caches_are_enabled()

    def reset_cache(self):
        """Reset both layers' key value caches."""
        self.layer.reset_cache()
        self.fusion_layer.reset_cache()

    def fusion_params(self) -> List[str]:
        """
        Return parameters of fusion layer.
        """
        fusion_params = [
            f"fusion_layer.{k}" for k, v in self.fusion_layer.named_parameters()
        ]
        return fusion_params

    def forward(self, x: torch.Tensor, **kwargs: Dict) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            **kwargs (Dict): all additional layer args

        Returns:
            Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]`

        """
        if self.fusion_first:
            x = self.fusion_layer(x, **kwargs)
            x = self.layer(x, **kwargs)
        else:
            x = self.layer(x, **kwargs)
            x = self.fusion_layer(x, **kwargs)
        return x


class FusionEmbedding(nn.Module):
    """Fusion embedding supports training additional special tokens while keeping
    the original embedding frozen. When fusing new models with a language model,
    there may be some additional tokens needed to support the fused language model. For
    example, adding a vision encoder might necessitate additional tokens like ``<|image|>``
    to indicate an images position in text and require learning an embedding for this token.
    The FusionEmbedding keeps the original embeddings frozen while learning a much smaller
    second embedding for the additional tokens. During forward this module routes
    the tokens to the appropriate embedding table.

    Use this as a drop-in replacement for :class:`torch.nn.Embedding` in your model.

    Example:
        >>> embedding = FusionEmbedding(vocab_size=100, fusion_vocab_size=10, embed_dim=128)
        >>> model = TransformerDecoder(tok_embeddings=embedding, ...)
        >>>
        >>> # Original model state_dict still works
        >>> model.load_state_dict(..., strict=False)

    .. note::
        This module assumes all tokens in the range [0, vocab_size) are part of the
        original embedding table and all new tokens in the range
        [vocab_size, vocab_size + fusion_vocab_size)

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
        key = prefix + "embedding.weight"
        new_key = prefix + "weight"
        destination[new_key] = destination[key]
        del destination[key]

    def _load_state_dict_hook(self, state_dict, prefix, *args, **kwargs):
        """Apply extra "embedding" prefix to the state_dict key to
        account for the FusionEmbedding wrapping.
        """
        if state_dict:
            key = prefix + "weight"
            new_key = prefix + "embedding.weight"
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input integer tensor with shape
                [batch_size x seq_length]

        Returns:
            Tensor: output tensor embedding with shape
                [batch_size x seq_length x embed_dim]`

        """
        bs, seq_len = input.size()
        vocab_size = self.embedding.num_embeddings

        mask = input < vocab_size
        # num_tokens = (input < vocab_size).sum()
        tokens = torch.masked_select(input, mask)
        # num_fusion_tokens = (input >= vocab_size).sum()
        fusion_tokens = torch.masked_select(input, ~mask) - vocab_size

        # [batch_size * num_tokens, embed_dim]
        embeds = self.embedding(tokens)
        # [batch_size * num_fusion_tokens, embed_dim]
        fusion_embeds = self.fusion_embedding(fusion_tokens)

        # [batch_size x seq_length x embed_dim]
        out = self._fused_embed(bs, seq_len)
        mask = mask.unsqueeze(-1).expand(bs, seq_len, self.dim)
        out = out.masked_scatter(mask, embeds)
        out = out.masked_scatter(~mask, fusion_embeds)
        return out
