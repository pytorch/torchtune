# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torchtune.modules import TransformerDecoder
from torchtune.modules.model_fusion._fusion_utils import get_fusion_params
from torchtune.modules.peft._utils import set_trainable_params


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


class DeepFusionModel(nn.Module):
    """DeepFusion is a type of fused model architecture where a pretrained encoder is combined
    with a pretrained decoder (LLM) in the internal decoder layers. This is a popular architecture for multimodal models, with
    a full overview available in `The Evolution of Multimodal Model Architectures <https://arxiv.org/abs/2405.17927>`_.
    A common deep fusion architecture is to fuse the encoder input into the decoder with interspersed cross-attention
    layers. This module makes no assumptions on how the encoder and decoder are fused; it simply
    passes in the encoder embeddings to the decoder and lets the decoder handle any fusion.

    This module has the same methods and forward signature as :class:`~torchtune.modules.TransformerDecoder` and can be used
    interchangeably where :class:`~torchtune.modules.TransformerDecoder` is. It combines the encoder with the decoder as a
    single module for checkpointing and finetuning. It is expected that the encoder and decoder
    are already defined with any extra learnable ``fusion_params``: learnable parameters to help
    adapt the pre-trained encoder to the pre-trained decoder.

    DeepFusionModel currently only supports a single encoder.

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
        >>> encoders = {"image": nn.Sequential(clip_vit_224(), projection_head)}
        >>>
        >>> # DeepFusionModel combines the encoder and decoder
        >>> model = DeepFusionModel(decoder, encoders)
        >>>
        >>> # Load full fused checkpoints (e.g. a Flamingo checkpoint)
        >>> model.load_state_dict(...)
        >>>
        >>> # Or load pretrained individual models (fusion_params are not loaded)
        >>> model.encoder.load_state_dict(..., strict=False)
        >>> model.decoder.load_state_dict(..., strict=False)
        >>>
        >>> # Forward pass
        >>> encoder_input = {"image": {...}}
        >>> output = model(tokens, mask, encoder_input, encoder_mask, input_pos)

    Args:
        decoder (TransformerDecoder): decoder module
        encoders (Dict[str, nn.Module]): dictionary mapping encoder name as a string to the encoder module.
        decoder_trainable (bool): whether to train or freeze the decoder. Default is False.
        encoders_trainable (Union[bool, Dict[str, bool]]): whether to train or freeze the encoder. Use a single
            boolean to set trainable for all encoders or a dictionary keyed by encoder names to specify trainable
            for each encoder individually. Encoder names should match with ``encoders``. Default is False.
        fusion_trainable (bool): whether to train the fusion parameters. Default is True.

    Raises:
        ValueError: if ``encoders`` and ``encoders_trainable`` keys do not match
        ValueError: if ``len(encoders) != 1``
    """

    def __init__(
        self,
        decoder: TransformerDecoder,
        encoders: Dict[str, nn.Module],
        *,
        decoder_trainable: bool = False,
        encoders_trainable: Union[bool, Dict[str, bool]] = False,
        fusion_trainable: bool = True,
    ):
        super().__init__()
        if (
            not isinstance(encoders_trainable, bool)
            and encoders.keys() != encoders_trainable.keys()
        ):
            raise ValueError(
                f"Found mismatched keys in encoders and encoders_trainable. Got {encoders.keys()} and {encoders_trainable.keys()}."
            )
        # Currently, only a single encoder is supported, so user can only
        # pass in a single key. When multiple encoders are
        # supported, this can be removed.
        if len(encoders.keys()) != 1:
            raise ValueError(
                f"DeepFusionModel only supports a single encoder. Got {len(encoders.keys())} encoders."
            )

        self.decoder = decoder
        self.encoders = nn.ModuleDict(encoders)
        self.encoders_trainable = (
            {k: encoders_trainable for k in self.encoders.keys()}
            if isinstance(encoders_trainable, bool)
            else encoders_trainable
        )

        trainable_params = set()
        for encoder, trainable in self.encoders_trainable.items():
            if trainable:
                trainable_params |= {
                    f"encoders.{encoder}.{n}"
                    for n, p in self.encoders[encoder].named_parameters()
                }
        if decoder_trainable:
            trainable_params |= {
                f"decoder.{n}" for n, p in self.decoder.named_parameters()
            }
        if fusion_trainable:
            trainable_params |= set(get_fusion_params(self))
        else:
            trainable_params -= set(get_fusion_params(self))
        set_trainable_params(self, trainable_params)

    def set_num_output_chunks(self, num_output_chunks: int) -> None:
        """Used to save memory in combination with :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss`.
        This should be called before the first forward pass, in the recipe."""
        self.decoder.set_num_output_chunks(num_output_chunks)

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: Optional[int] = None,
        decoder_max_seq_len: Optional[int] = None,
    ):
        """
        Sets up key-value attention caches for inference for ``self.decoder``.
        For each layer in ``self.decoder.layers``:
        - :class:`torchtune.modules.TransformerSelfAttentionLayer` will use ``decoder_max_seq_len``.
        - :class:`torchtune.modules.TransformerCrossAttentionLayer` will use ``encoder_max_seq_len``.
        - :class:`torchtune.modules.fusion.FusionLayer` will use both ``decoder_max_seq_len`` and ``encoder_max_seq_len``.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (Optional[int]): maximum encoder cache sequence length.
            decoder_max_seq_len (Optional[int]): maximum decoder cache sequence length.
        """
        self.decoder.setup_caches(
            batch_size,
            dtype,
            encoder_max_seq_len=encoder_max_seq_len,
            decoder_max_seq_len=decoder_max_seq_len,
        )

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup. This means ``setup_caches`` has been called, and
        the relevant attention modules in the model have created their ``KVCache``.
        """
        return self.decoder.caches_are_setup()

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches are enabled. Once KV-caches have been setup, the relevant
        attention modules will be "enabled" and all forward passes will update the caches. This behaviour
        can be disabled without altering the state of the KV-caches by "disabling" the KV-caches
        using :func:`~torchtune.modules.common_utils.disable_kv_cache`, upon which ``caches_are_enabled`` would return False.
        """
        return self.decoder.caches_are_enabled()

    def reset_caches(self):
        """
        Resets KV-cache buffers on relevant attention modules to zero, and reset cache positions to zero,
        without deleting or reallocating cache tensors.
        """
        self.decoder.reset_caches()

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[Dict[str, Dict[str, Any]]] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            mask (Optional[torch.Tensor]): Optional boolean tensor which contains the attention mask
                with shape ``[b x s x s]``. This is applied after the query-key multiplication and
                before the softmax. A value of True in row i and column j means token i attends
                to token j. A value of False means token i does not attend to token j. If no
                mask is specified, a causal mask is used by default. Default is None.
            encoder_input (Optional[Dict[str, Dict[str, Any]]]): Optional input kwargs for the encoders. Must be
                keyed by encoder name and match the keys of ``encoders``
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position i,j means token i can attend
                to embedding j in the decoder. Mask has shape ``[b x s x s_e]``. Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            Tensor: output tensor with shape ``[b x s x v]`` or a list of layer \
                output tensors defined by ``output_hidden_states`` with the \
                final output tensor appended to the list.

        Raises:
            ValueError: if ``encoder_input`` keys do not match ``encoders`` keys

        Notation used for tensor shapes:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        if encoder_input is not None and encoder_input.keys() != self.encoders.keys():
            raise ValueError(
                f"Found mismatched keys in encoder_input and instantiated encoders. "
                f"Got {encoder_input.keys()}, expected {self.encoders.keys()}."
            )
        # During decoding, encoder_input will only be provided
        # for new inputs. Previous encoder outputs are cached
        # in the decoder cache.
        encoder_embed = None
        if encoder_input is not None:
            encoder_embed = {
                encoder: self.encoders[encoder](**encoder_input[encoder])
                for encoder in encoder_input
            }

        # Currently, only a single encoder is supported, so we need
        # to get the encoder key manually. When multiple encoders are
        # supported, this can be removed.
        decoder_encoder_input = (
            list(encoder_embed.values())[0] if encoder_embed is not None else None
        )

        output = self.decoder(
            tokens=tokens,
            mask=mask,
            encoder_input=decoder_encoder_input,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
        )
        return output


class EarlyFusionModel(nn.Module):
    """EarlyFusion is a type of fused model architecture where pretrained encoder(s) are combined
    with a pretrained decoder (LLM) at the model input and not in internal layers. This is a popular architecture
    for multimodal models, with a full overview available in `The Evolution of Multimodal Model Architectures
    <https://arxiv.org/abs/2405.17927>`_. This module works both for decoders in which the encoder tokens are
    inside the vocab and outside the vocab.

    This module has the same methods and forward signature as :class:`~torchtune.modules.TransformerDecoder` and can be used
    interchangeably where :class:`~torchtune.modules.TransformerDecoder` is. It combines the encoders with the decoder as a
    single module for checkpointing and finetuning. It is expected that the encoders and decoder
    are already defined with any extra learnable ``fusion_params``: learnable parameters to help
    adapt the pre-trained encoders to the pre-trained decoder.

    You can pass in multiple encoders as a dictionary into ``encoders``.

    Note: Once the decoder is wrapped in this module, the decoder's ``tok_embeddings`` module is moved
    to the parent EarlyFusionModel's ``tok_embeddings``. You should not forward pass the decoder individually.
    Instead, use EarlyFusionModel's forward pass with ``encoder_input=None`` to get decoder-only outputs.
    State dicts will automatically be updated on save and load to account for this change.

    Example:
        >>> # decoder is a text-only TransformerDecoder (e.g. llama3_8b) with no modifications
        >>> decoder = llama3_8b()
        >>>
        >>> # encoder is pre-trained encoder (e.g. clip_vit_224) with an added projection head
        >>> projection_head = FeedForward(...)
        >>> register_fusion_module(projection_head))
        >>> encoders = {"image": nn.Sequential(clip_vit_224(), projection_head)}
        >>>
        >>> # EarlyFusionModel combines the encoder and decoder
        >>> model = EarlyFusionModel(decoder, encoders, encoder_tokens={"image": 128256})
        >>>
        >>> # Load full fused checkpoints
        >>> model.load_state_dict(...)
        >>>
        >>> # Forward pass
        >>> encoder_input = {"image": {...}}
        >>> output = model(tokens, mask=mask, encoder_input=encoder_input, encoder_mask=encoder_mask, input_pos=input_pos)
        >>>
        >>> # Forward pass decoder only
        >>> output = model(tokens, mask=mask, input_pos=input_pos)

    Args:
        decoder (TransformerDecoder): decoder module
        encoders (Dict[str, nn.Module]): dictionary mapping encoder name as a string to the encoder module.
        encoder_tokens (Dict[str, int]): dictionary mapping encoder name to special token ID indicating where
            in the text sequence the encoder embedding outputs should be injected.
        decoder_trainable (bool): whether to train or freeze the decoder. Default is False.
        encoders_trainable (Union[bool, Dict[str, bool]]): whether to train or freeze the encoder. Use a single
            boolean to set trainable for all encoders or a dictionary keyed by encoder names to specify trainable
            for each encoder individually. Encoder names should match with ``encoders``. Default is False.
        fusion_trainable (bool): whether to train the fusion parameters. Default is True.

    Raises:
        ValueError: if ``encoders`` and ``encoders_trainable`` keys do not match
    """

    def __init__(
        self,
        decoder: TransformerDecoder,
        encoders: Dict[str, nn.Module],
        encoder_tokens: Dict[str, int],
        decoder_trainable: bool = False,
        encoders_trainable: Union[bool, Dict[str, bool]] = False,
        fusion_trainable: bool = True,
    ):
        super().__init__()
        if encoders.keys() != encoder_tokens.keys() or (
            not isinstance(encoders_trainable, bool)
            and encoders.keys() != encoders_trainable.keys()
        ):
            raise ValueError(
                f"Found mismatched keys in encoders, encoder_tokens, and/or encoders_trainable. Expected {encoders.keys()}"
            )

        self.decoder = decoder
        self.encoders = nn.ModuleDict(encoders)
        self.encoder_tokens = encoder_tokens
        self.encoders_trainable = (
            {k: encoders_trainable for k in self.encoders.keys()}
            if isinstance(encoders_trainable, bool)
            else encoders_trainable
        )

        # A little surgery in the decoder to give the
        # fusion module access to control the embeddings
        # The alternative is to pass a special tok_embeddings
        # module into TransformerDecoder builder that does the
        # merging there
        self.tok_embeddings = decoder.tok_embeddings
        decoder.tok_embeddings = nn.Identity()

        self._register_state_dict_hook(self._state_dict_hook)
        self.register_load_state_dict_pre_hook(self._load_state_dict_hook)

        trainable_params = set()
        for encoder, trainable in self.encoders_trainable.items():
            if trainable:
                trainable_params |= {
                    f"encoders.{encoder}.{n}"
                    for n, p in self.encoders[encoder].named_parameters()
                }
        if decoder_trainable:
            trainable_params |= {
                f"decoder.{n}" for n, p in self.decoder.named_parameters()
            }
            trainable_params |= {
                f"tok_embeddings.{n}" for n, p in self.tok_embeddings.named_parameters()
            }
        if fusion_trainable:
            trainable_params |= set(get_fusion_params(self))
        else:
            trainable_params -= set(get_fusion_params(self))

        set_trainable_params(self, trainable_params)

    @staticmethod
    def _state_dict_hook(module, state_dict, *args, **kwargs):
        """
        Keep tok_embeddings inside of decoder state_dict

        [!Note] This update changes the order of the OrderedDict
        """
        for n, p in module.tok_embeddings.named_parameters():
            state_dict[f"decoder.tok_embeddings.{n}"] = p
            del state_dict[f"tok_embeddings.{n}"]

    @staticmethod
    def _load_state_dict_hook(module, state_dict, *args, **kwargs):
        """Undo the change from _state_dict_hook"""
        old_keys = list(state_dict.keys())
        for key in old_keys:
            if key.startswith("decoder.tok_embeddings"):
                state_dict[key[len("decoder.") :]] = state_dict[key]
                del state_dict[key]

    def set_num_output_chunks(self, num_output_chunks: int) -> None:
        """Used to save memory in combination with :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss`.
        This should be called before the first forward pass, in the recipe."""
        self.decoder.set_num_output_chunks(num_output_chunks)

    def setup_caches(self, batch_size: int, dtype: torch.dtype) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
        """
        self.decoder.setup_caches(batch_size, dtype)

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup. This means ``setup_caches`` has been called, and
        the relevant attention modules in the model have created their ``KVCache``.
        """
        return self.decoder.caches_are_setup()

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches are enabled. Once KV-caches have been setup, the relevant
        attention modules will be "enabled" and all forward passes will update the caches. This behaviour
        can be disabled without altering the state of the KV-caches by "disabling" the KV-caches
        using :func:`~torchtune.modules.common_utils.disable_kv_cache`, upon which ``caches_are_enabled`` would return False.
        """
        return self.decoder.caches_are_enabled()

    def reset_caches(self):
        """Reset the key value caches."""
        self.decoder.reset_caches()

    def _decoder_embed(self, tokens) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed the text-only tokens with the decoder's tok_embeddings"""
        encoder_token_ids = torch.tensor(list(self.encoder_tokens.values()))
        # [bsz, seq_len], True indicates the token is not an encoder special token
        is_text = ~torch.isin(tokens, encoder_token_ids)
        text_tokens = torch.masked_select(tokens, is_text)
        # [num_text, embed_dim]
        text_embeds = self.tok_embeddings(text_tokens)
        return is_text, text_embeds

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[Dict[str, Dict[str, Any]]] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: Dict[str, Any],  # no need for encoder_mask
    ) -> torch.Tensor:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            mask (Optional[torch.Tensor]): Optional boolean tensor which contains the attention mask
                with shape ``[b x s x s]``. This is applied after the query-key multiplication and
                before the softmax. A value of True in row i and column j means token i attends
                to token j. A value of False means token i does not attend to token j. If no
                mask is specified, a causal mask is used by default. Default is None.
            encoder_input (Optional[Dict[str, Dict[str, Any]]]): Optional input kwargs for the encoders. Must be
                keyed by encoder name and match the keys of ``encoders``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.
            **kwargs (Dict[str, Any]): additional keyword arguments. This is solely used to match the
                :class:`~torchtune.modules.TransformerDecoder` forward and does not have any effect.

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            torch.Tensor: output tensor with shape ``[b x s x v]`` or a list of layer \
                output tensors defined by ``output_hidden_states`` with the \
                final output tensor appended to the list.

        Raises:
            ValueError: if ``encoder_input`` keys do not match ``encoders`` keys

        Notation used for tensor shapes:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        if encoder_input is not None and encoder_input.keys() != self.encoders.keys():
            raise ValueError(
                f"Found mismatched keys in encoder_input and instantiated encoders. "
                f"Got {encoder_input.keys()}, expected {self.encoders.keys()}."
            )

        bsz, seq_len = tokens.shape
        # is_text: [bsz, seq_len], text_embeds: [num_text, embed_dim]
        is_text, text_embeds = self._decoder_embed(tokens)
        embed_dim = text_embeds.shape[-1]

        # Holds the final embedding vector
        fused_embeds = torch.empty(
            bsz, seq_len, embed_dim, dtype=text_embeds.dtype, device=text_embeds.device
        )
        # Place the text-only embeddings
        fused_embeds = fused_embeds.masked_scatter(is_text.unsqueeze(-1), text_embeds)

        for encoder, inp in (encoder_input or {}).items():
            # [bsz, num_encoder_tokens, embed_dim]
            encoder_embeds = self.encoders[encoder](**inp)
            # [bsz * num_encoder_tokens, embed_dim]
            encoder_embeds = encoder_embeds.view(-1, embed_dim)
            # [bsz, seq_len, 1]
            encoder_mask = (tokens == self.encoder_tokens[encoder]).unsqueeze(-1)
            # At locations where encoder token is found, replace with encoder embedding
            fused_embeds = fused_embeds.masked_scatter(encoder_mask, encoder_embeds)

        output = self.decoder(fused_embeds, mask=mask, input_pos=input_pos)
        return output
