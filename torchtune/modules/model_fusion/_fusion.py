# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Union

import torch
from torch import nn, Tensor
from torchtune.modules import TransformerDecoder


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

    def _state_dict_hook(self, state_dict, *args, **kwargs):
        """Remove "layer" from the original layer in the state_dict
        name. This keeps the orginal state dict name for the layer
        from before fusing with the FusionLayer.

        [!Note] This update changes the order of the OrderedDict
        """
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith("layer"):
                new_key = key.replace("layer.", "")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

    def _load_state_dict_hook(self, state_dict, *args, **kwargs):
        """Apply extra "layer" prefix to the state_dict key to
        account for the FusionLayer wrapping.
        """
        keys = list(state_dict.keys())
        for key in keys:
            if not key.startswith("fusion_layer"):
                new_key = "layer." + key
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

    def setup_cache(self, batch_size: int, dtype: torch.dtype) -> None:
        """Setup key value cache for both layers.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
        """
        self.layer.setup_cache(batch_size, dtype)
        self.fusion_layer.setup_cache(batch_size, dtype)

    @property
    def cache_enabled(self) -> bool:
        """Check if the key value caches are setup."""
        return self.layer.cache_enabled

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

    def forward(self, x: Tensor, **kwargs: Dict) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
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

    Use this as a drop-in replacement for `nn.Embedding` in your model.

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
        # num_tokens = (input < vocab_size).sum()
        tokens = torch.masked_select(input, mask)
        # num_fusion_tokens = (input >= vocab_size).sum()
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


class DeepFusionModel(nn.Module):
    """DeepFusion is a type of fused model architecture where a pretrained encoder is combined
    with a pretrained decoder (LLM). This is a popular architecture for multimodal models, with
    a full overview available in `The Evolution of Multimodal Model Architectures <https://arxiv.org/abs/2405.17927>`_.

    This module has the same methods and forward signature as :class:`~torchtune.modules.TransformerDecoder` and can be used
    interchangeably where :class:`~torchtune.modules.TransformerDecoder` is. It combines the encoder with the decoder as a
    single module for checkpointing and finetuning. It is expected that the encoder and decoder
    are already defined with any extra learnable ``fusion_params``; learnable parameters to help
    adapt the pre-trained encoder to the pre-trained decoder.

    Example:
        >>> # decoder is a TransformerDecoder (e.g. llama3_8b) with fused cross attention layers
        >>> embed = FusionEmbedding(...)
        >>> layer = FusionLayer(
        ...     layer=TransformerSelfAttentionLayer(...),
        ...     fusion_layer=TransformerCrossAttentionLayer(...),
        ... )
        >>> decoder = TransformerDecoder(tok_embeddings=embed, layers=layer, num_layers=32, ...)
        >>>
        >>> # encoder is pre-trained encoder (e.g. clip_vit_224) with an added projection head
        >>> projection_head = FeedForward(...)
        >>> register_fusion_module(projection_head))
        >>> encoder = nn.Sequential(clip_vit_224(), projection_head)
        >>>
        >>> # DeepFusionModel combines the encoder and decoder
        >>> model = DeepFusionModel(decoder, encoder)
        >>>
        >>> # Load full fused checkpoints (e.g. a Flamingo checkpoint)
        >>> model.load_state_dict(...)
        >>>
        >>> # Or load pretrained individual models (fusion_params are not loaded)
        >>> model.encoder.load_state_dict(..., strict=False)
        >>> model.decoder.load_state_dict(..., strict=False)
        >>>
        >>> # Forward pass
        >>> output = model(tokens, mask, encoder_input, encoder_mask, input_pos)

    Args:
        decoder (TransformerDecoder): decoder module
        encoder (nn.Module): encoder module
    """

    def __init__(
        self,
        decoder: TransformerDecoder,
        encoder: nn.Module,
    ):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder

    def setup_caches(self, batch_size: int, dtype: torch.dtype) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
        """
        self.decoder.setup_caches(batch_size, dtype)

    def caches_are_enabled(self) -> bool:
        """Check if the key value caches are setup."""
        return self.decoder.caches_are_enabled()

    def reset_caches(self):
        """Reset the key value caches."""
        self.decoder.reset_caches()

    def forward(
        self,
        tokens: Tensor,
        *,
        mask: Optional[Tensor] = None,
        encoder_input: Optional[Dict] = None,
        encoder_mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ) -> Union[Tensor, List[Tensor]]:
        """
        Args:
            tokens (Tensor): input tensor with shape [b x s]
            mask (Optional[Tensor]): Optional boolean tensor which contains the attention mask
                with shape [b x s x s]. This is applied after the query-key multiplication and
                before the softmax. A value of True in row i and column j means token i attends
                to token j. A value of False means token i does not attend to token j. If no
                mask is specified, a causal mask is used by default. Default is None.
            encoder_input (Optional[Dict]): Optional input for the encoder.
            encoder_mask (Optional[Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position i,j means token i can attend
                to embedding j in the decoder. Mask has shape [b x s x s_e]. Default is None.
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            Tensor: output tensor with shape [b x s x v] or a list of layer
                output tensors defined by ``output_hidden_states`` with the
                final output tensor appended to the list.

        Notation used for tensor shapes:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        # During decoding, encoder_input will only be provided
        # for new inputs. Previous encoder outputs are cached
        # in the decoder cache.
        encoder_embed = None
        if encoder_input is not None:
            encoder_embed = self.encoder(**encoder_input)

        # input_pos specifies the sequence position of the provided tokens
        # we slice the encoder_mask to only include those same positions
        if input_pos is not None and encoder_mask is not None:
            encoder_mask = encoder_mask[:, input_pos]
        output = self.decoder(
            tokens=tokens,
            mask=mask,
            encoder_input=encoder_embed,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
        )
        return output
