# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torchtune.models.clip._component_builders import clip_text_encoder
from torchtune.models.clip._text_encoder import CLIPTextEncoder
from torchtune.models.clip._tokenizer import CLIPTokenizer
from torchtune.models.clip._transform import CLIPImageTransform


def clip_tokenizer(
    path: str,
    max_seq_len: int = 77,
    truncate: bool = True,
) -> CLIPTokenizer:
    """
    Builder for the CLIP text tokenizer.

    Args:
        path (str): Path to the CLIP merges file
        max_seq_len (bool): Context length. Default: 77
        truncate (bool): Truncate the token sequence if it exceeds max_seq_len (otherwise raises AssertionError)
            Default: True

    Returns:
        CLIPTokenizer: Instantiation of the CLIP text tokenizer
    """
    return CLIPTokenizer(path, max_seq_len=max_seq_len, truncate=truncate)


def clip_text_vit_large_patch14() -> CLIPTextEncoder:
    """
    Builder for the CLIP text encoder for CLIP-ViT-L/14.

    CLIP is a model that encodes text and images into a shared vector space.
    Blog post: https://openai.com/index/clip/
    Paper: https://arxiv.org/abs/2103.00020

    Returns:
        CLIPTextEncoder: Instantiation of the CLIP text encoder
    """
    return clip_text_encoder(
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        vocab_size=49408,
        max_seq_len=77,
        norm_eps=1e-5,
    )


def clip_vit_224_transform():
    image_transform = CLIPImageTransform(
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        tile_size=224,
        possible_resolutions=None,
        max_num_tiles=1,
        resample="bilinear",
        resize_to_max_canvas=True,
    )

    return image_transform
