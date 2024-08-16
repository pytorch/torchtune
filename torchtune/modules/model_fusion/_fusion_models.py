# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Union

import torch
from torch import nn, Tensor
from torchtune.modules import TransformerDecoder


class DeepFusionModel(nn.Module):
    """DeepFusion is a type of fused model architecture where a pretrained encoder is combined
    with a pretrained decoder (LLM). This is a popular architecture for multimodal models, with
    a full overview available in `The Evolution of Multimodal Model Architectures <https://arxiv.org/abs/2405.17927>`_.

    This module has the same methods and forward signature as :class:`~torchtune.modules.TransformerDecoder` and can be used
    interchangeably where :class:`~torchtune.modules.TransformerDecoder` is. It combines the encoder with the decoder as a
    single module for checkpointing and finetuning. It is expected that the encoder and decoder
    are already defined with any extra learnable ``fusion_params``; learnable parameters to help
    adapt the pre-trained encoder to the pre-trained decoder.

    Example::
        >>> model = DeepFusionModel(LLama3(), CLIP())

        # Load full checkpoints
        >>> model.load_state_dict(...)

        # Or load pretrained individual models
        >>> model.encoder.load_state_dict(...)
        >>> model.decoder.load_state_dict(...)

        # Forward pass
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
            encoder_input (Optional[Dict]): Optional input for the encoder
            encoder_mask (Optional[Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position i,j means token i can attend
                to embedding j in the decoder. Mask has shape
                [batch_size x token_seq_len x encoder_seq_len]. Default is None.
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Union[Tensor, List[Tensor]]: output tensor with shape [b x s x v] or a list of layer
                output tensors defined by ``output_hidden_states`` with the final output tensor
                appended to the list.
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
