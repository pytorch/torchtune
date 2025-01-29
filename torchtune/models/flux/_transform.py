# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Mapping, Tuple

import torch
from torch import Tensor

from torchtune.models.clip import clip_tokenizer
from torchtune.models.flux._util import get_t5_max_seq_len
from torchtune.models.t5 import t5_tokenizer
from torchtune.modules.transforms import Transform
from torchvision import transforms


class FluxTransform(Transform):
    """
    Transform general text-to-image data to Flux-specific data.

    Args:
        flux_model_name (str): "FLUX.1-dev" or "FLUX.1-schnell" (affects the T5 max seq len)
        img_width (int): Resize images to this width.
        img_height (int): Resize images to this height.
        clip_tokenizer_path (str): Path to the CLIP tokenizer `merges.txt` file.
        t5_tokenizer_path (str): Path to the T5 tokenizer `spiece.model` file.
        truncate_text (bool): Whether to truncate the tokenized text if longer than the max seq len.
            If false, raises an `AssertionError` if the text is too long.
            Default: False
    """

    def __init__(
        self,
        flux_model_name: str,
        img_width: int,
        img_height: int,
        clip_tokenizer_path: str,
        t5_tokenizer_path: str,
        truncate_text: bool = False,
    ):
        self._clip_tokenizer = clip_tokenizer(
            clip_tokenizer_path, truncate=truncate_text
        )

        self._t5_tokenizer = t5_tokenizer(
            t5_tokenizer_path, max_seq_len=get_t5_max_seq_len(flux_model_name)
        )

        self._img_transform = transforms.Compose(
            [
                transforms.Resize(max(img_width, img_height)),
                transforms.CenterCrop((img_width, img_height)),
                transforms.ToTensor(),
                _FluxImgNormalize(),
            ]
        )

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Transform a general text-to-image row to Flux-specific row.

        Args:
            sample (Mapping[str, Any]): PIL image and text

        Returns:
            Mapping[str, Any]: Image tensor and CLIP/T5 tokens
        """
        img = self._img_transform(sample["image"])
        clip_tokens, t5_tokens = self.tokenize(sample["text"])
        return {
            "image": img,
            "clip_tokens": clip_tokens,
            "t5_tokens": t5_tokens,
        }

    def tokenize(self, text: str) -> Tuple[Tensor, Tensor]:
        """
        Tokenize a string using the CLIP and T5 tokenizers.

        Args:
            text (str): the string

        Returns:
            Tuple[Tensor, Tensor]: tuple of (clip_tokens, t5_tokens)
        """
        clip_tokens = self._tokenize(text, self._clip_tokenizer)
        t5_tokens = self._tokenize(text, self._t5_tokenizer)
        return clip_tokens, t5_tokens

    def _tokenize(self, text, tokenizer):
        tensor = torch.full(
            (tokenizer.max_seq_len,),
            tokenizer.pad_id,
            dtype=torch.int,
        )
        tokens = tokenizer.encode(text)
        tensor[: len(tokens)] = torch.tensor(tokens)
        return tensor


class _FluxImgNormalize:
    def __call__(self, x):
        return (x * 2.0) - 1.0
