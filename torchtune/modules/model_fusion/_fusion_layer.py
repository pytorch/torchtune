# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch
from torch import nn, Tensor


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
