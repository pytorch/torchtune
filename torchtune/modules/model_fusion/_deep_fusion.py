# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from torchtune.modules import TransformerDecoder
from torchtune.modules.model_fusion._fusion_utils import get_fusion_params
from torchtune.modules.peft._utils import set_trainable_params
from torchtune.utils import get_logger, log_once

logger = get_logger("DEBUG")


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
        >>> encoder = nn.Sequential(clip_vit_224(), projection_head)
        >>>
        >>> # DeepFusionModel combines the encoder and decoder
        >>> model = DeepFusionModel(decoder, encoder)
        >>>
        >>> # Load full fused checkpoints (e.g. a Llama3.2 Vision checkpoint)
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
        decoder_trainable (bool): whether to train or freeze the decoder. Default is False.
        encoder_trainable (bool): whether to train or freeze the encoder. Default is False.
        fusion_trainable (bool): whether to train the fusion parameters. Default is True.

    """

    def __init__(
        self,
        decoder: TransformerDecoder,
        encoder: nn.Module,
        *,
        decoder_trainable: bool = False,
        encoder_trainable: bool = False,
        fusion_trainable: bool = True,
    ):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder

        trainable_params = set()
        if encoder_trainable:
            trainable_params |= {
                f"encoder.{n}" for n, p in self.encoder.named_parameters()
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

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[Dict] = None,
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
            encoder_input (Optional[Dict]): Optional input for the encoder.
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

        output = self.decoder(
            tokens=tokens,
            mask=mask,
            encoder_input=encoder_embed,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
        )
        return output
