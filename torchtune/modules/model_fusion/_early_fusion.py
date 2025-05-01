# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torchtune.modules import TransformerDecoder
from torchtune.modules.model_fusion._fusion_utils import get_fusion_params
from torchtune.modules.peft._utils import set_trainable_params


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
        msg = (
            "'set_num_output_chunks' is deprecated and will be removed in future versions. "
            "Please use self.skip_linear_projection=True and do the chunking in your loss instead, "
            "e.g. loss(weight, input, label)."
        )
        log_once(logger=logger, msg=msg, level=logging.WARNING)
        self.decoder.set_num_output_chunks(num_output_chunks)

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: Optional[int] = None,
        decoder_max_seq_len: Optional[int] = None,
    ) -> None:
        """
        Setup key value caches for attention calculation.
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
        """Reset the key value caches."""
        self.decoder.reset_caches()

    @property
    def linear_projection_weight(self) -> torch.Tensor:
        """Returns the output weight matrix. Useful when a finer control of the output projection is needed,
        for example when using a custom loss function or when interested in applying it to only some tokens.
        """
        return self.decoder.linear_projection_weight

    @property
    def skip_linear_projection(self) -> bool:
        """Returns whether to skip output layer projection and return hidden states instead."""
        return self.decoder.skip_linear_projection

    @skip_linear_projection.setter
    def skip_linear_projection(self, skip: bool) -> None:
        """Set whether to skip output layer projection and return hidden states instead."""
        self.decoder.skip_linear_projection = skip

    def _decoder_embed(self, tokens) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed the text-only tokens with the decoder's tok_embeddings"""
        encoder_token_ids = torch.tensor(
            list(self.encoder_tokens.values()), device=tokens.device
        )
        # [bsz, seq_len], True indicates the token is not an encoder special token
        is_text = ~torch.isin(tokens, encoder_token_ids)
        text_tokens = torch.masked_select(tokens, is_text)
        # [num_text, embed_dim]

        text_embeds = self.decoder.tok_embeddings(text_tokens)
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
        Note: This module assumes that there will be enough encoder inputs (i.e., total number of images in the batch)
        for the number of encoder tokens in the batch.

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
        if encoder_input is not None:
            if any(key not in self.encoders.keys() for key in encoder_input):
                raise ValueError(
                    f"Found missing keys of encoder_input in instantiated encoders. "
                    f"Got {self.encoders.keys()}, expected {encoder_input.keys()}."
                )

        bsz, seq_len = tokens.shape
        # is_text: [bsz, seq_len], text_embeds: [num_text, embed_dim]
        is_text, text_embeds = self._decoder_embed(tokens)
        embed_dim = text_embeds.shape[-1]

        # Holds the final embedding vector
        fused_embeds = torch.empty(
            bsz, seq_len, embed_dim, dtype=text_embeds.dtype, device=text_embeds.device
        )
        # Place the text-only embeddings, fused_embeds: [bsz, seq_len, embed_dim]
        fused_embeds = fused_embeds.masked_scatter(is_text.unsqueeze(-1), text_embeds)

        encoder_input = encoder_input or {}
        for encoder, inp in encoder_input.items():
            # [bsz, num_encoder_tokens, embed_dim]
            encoder_embeds = self.encoders[encoder](**inp)
            # [bsz * num_encoder_tokens, embed_dim]
            encoder_embeds = encoder_embeds.view(-1, embed_dim)
            # [bsz, seq_len, 1]
            encoder_mask = (tokens == self.encoder_tokens[encoder]).unsqueeze(-1)
            # At locations where encoder token is found, replace with encoder embedding
            # Note: the encoder mask will account for the embeddings padding since we only
            # add encoder tokens to text tokens for the non-padding part.
            fused_embeds = fused_embeds.masked_scatter(encoder_mask, encoder_embeds)

        output = self.decoder(
            tokens=None, mask=mask, input_pos=input_pos, input_embeds=fused_embeds
        )
        return output
