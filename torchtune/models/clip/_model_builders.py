# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torchtune.models.clip._transform import CLIPImageTransform
from pathlib import Path

from torchtune.utils._download import download_file, TORCHTUNE_LOCAL_CACHE_FOLDER
from torchtune.models.clip._tokenizer import CLIPTokenizer
from torchtune.models.clip._text_encoder import CLIPTextEncoder


CLIP_VOCAB_URL = 'https://github.com/openai/CLIP/raw/refs/heads/main/clip/bpe_simple_vocab_16e6.txt.gz'


def clip_tokenizer(vocab_path: Path = TORCHTUNE_LOCAL_CACHE_FOLDER / 'clip_vocab.txt.gz', download_if_missing: bool = True, max_seq_len: int = 77, truncate: bool = True) -> CLIPTokenizer:
    """
    Builder for the CLIP text tokenizer.

    Args:
        vocab_path (pathlib.Path): Path to the CLIP vocab file
            Default: '~/.cache/torchtune/clip_vocab.txt.gz'
        download_if_missing (bool): Download the vocab file if it's not found
            Default: True
        max_seq_len (bool): Context length
            Default: 77
        truncate (bool): Truncate the token sequence if it exceeds max_seq_len (otherwise raises AssertionError)
            Default: True

    Returns:
        CLIPTokenizer: Instantiation of the CLIP text tokenizer
    """
    if not vocab_path.exists():
        assert download_if_missing, f'Missing CLIP tokenizer vocab: {vocab_path}'
        download_file(CLIP_VOCAB_URL, vocab_path)
    
    return CLIPTokenizer(vocab_path, max_seq_len=max_seq_len, truncate=truncate)


def clip_text_encoder_large() -> CLIPTextEncoder:
    """
    Builder for the CLIP text encoder for CLIP-ViT-L/14.

    Returns:
        CLIPTextEncoder: Instantiation of the CLIP text encoder
    """
    return CLIPTextEncoder(
        vocab_size=49408,
        max_seq_len=77,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
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
