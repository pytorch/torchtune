# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List

import torch

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX


def padded_collate(
    batch: List[Dict[str, List[int]]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> Dict[str, torch.Tensor]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors.

    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries containing input, label pairs.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.

    Returns:
        Dict[str, torch.Tensor]: Collated input and label tensors.

    Example:
        >>> token_pairs = [
        >>>    {"tokens": [1, 2, 3], "labels": [4, 5, 6]},
        >>>    {"tokens": [7,], "labels": [10,]},
        >>> ]
        >>> collated = padded_collate(
        >>>    batch=token_pairs,
        >>>    padding_idx=padding_idx,
        >>>    ignore_idx=ignore_idx,
        >>> )
        >>> collated["tokens"]
        >>> tensor([[1, 2, 3], [7, 0, 0]])
        >>> collated["labels"]
        >>> tensor([[4, 5, 6], [10, -100, -100]])
    """
    input_ids = pad_sequence(
        [torch.tensor(x["tokens"]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    labels = pad_sequence(
        [torch.tensor(x["labels"]) for x in batch],
        batch_first=True,
        padding_value=ignore_idx,
    )

    input_ids_seq_len = input_ids.shape[-1]
    labels_seq_len = labels.shape[-1]

    # Hack to pad correctly and not use max_seq_len, which is costly
    if input_ids_seq_len > labels_seq_len:
        labels = F.pad(
            labels, (0, input_ids_seq_len - labels_seq_len), value=ignore_idx
        )
    elif labels_seq_len > input_ids_seq_len:
        input_ids = F.pad(
            input_ids,
            (0, labels_seq_len - input_ids_seq_len),
            value=padding_idx,
        )
    return {"tokens": input_ids.long(), "labels": labels.long()}


def padded_collate_tiled_images_with_cross_attention(
    batch: List[Dict[str, Any]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> Dict[str, torch.Tensor]:
    """Pad a batch of text sequences, tiled image tensors, aspect ratios,
    and cross attention masks.

    ``batch`` is expected to be a list of sample dicts containing the following:
        - "tokens": List[int] of length text_seq_len, varies across samples
        - "labels": List[int] of length text_seq_len, varies across samples
        - "images": List[Tensor], each with shape (n_tiles, c, h, w)
        - "encoder_mask": List[Tensor], each with shape (text_seq_len, image_seq_len)
        - "aspect_ratio": List[Tensor], each with shape (h_ratio, w_ratio)

    This collater does the following:
        (1) Pad text sequence and encoder mask to the longest sequence length in the batch
        (2) Pad image tensors in the tile dimension with zeros to the largest number
            of tiles in the batch
        (3) Add empty images of zeros to samples up to max number of images in the batch
        (4) Pad aspect ratios with (1,1) for all added padding images

    Args:
        batch (List[Dict[str, Any]]): A list of sample dicts containing tokens,
            labels, images, encoder_mask, and aspect_ratio.
        padding_idx (int): Padding index for input token ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.

    Returns:
        Dict[str, Tensor]: Collated tokens, labels, images, encoder_mask, aspect_ratio tensors.

    Example:
        >>> image_id = 1
        >>> tokens_per_tile = 5
        >>> c, h, w = 1, 1, 1
        >>> batch = [
        ...     {
        ...         "tokens": [1, 2, 1, 3], "labels": [4, 5, 6, 7],
        ...         "images": [torch.ones(2, c, h, w), torch.ones(3, c, h, w)],
        ...         "encoder_mask": [torch.ones(4, 5 * 2), torch.ones(4, 5 * 3)],
        ...         "aspect_ratio": [torch.tensor([1, 2]), torch.tensor([1, 2])],
        ...     },
        ...     {
        ...         "tokens": [1, 4], "labels": [8, 9],
        ...         "images": [torch.ones(4, c, h, w)],
        ...         "encoder_mask": [torch.ones(2, 5 * 4)],
        ...         "aspect_ratio": [torch.tensor([1, 2])],
        ...     },
        ... ]
        >>> model_inputs = padded_collate_vision_text(batch=batch)
        >>> print(model_inputs["tokens"])
        tensor([[1, 2, 1, 3], [1, 4, 0, 0]])
        >>> print(model_inputs["labels"])
        tensor([[4, 5, 6, 7], [8, 9, -100, -100]])
        >>> print(model_inputs["images"].shape)
        torch.Size([2, 2, 4, 1, 1, 1])
        >>> print(model_inputs["encoder_mask"].shape)
        torch.Size([2, 2, 4, 20])
        >>> print(model_inputs["aspect_ratio"].shape)
        torch.Size([2, 2, 2])
        >>> print(model_inputs["images"][0, 0, ...])  # Image with two tiles got padded to four
        tensor([[[[1.]]], [[[1.]]], [[[0.]]], [[[0.]]]])
        >>> print(model_inputs["images"][0, 1, ...])  # Image with three tiles got padded to four
        tensor([[[[1.]]], [[[1.]]], [[[1.]]], [[[0.]]]])
        >>> print(model_inputs["images"][1, 0, ...])  # Image with four tiles did not get padded
        tensor([[[[1.]]], [[[1.]]], [[[1.]]], [[[1.]]]])
        >>> print(model_inputs["images"][1, 1, ...])  # Extra padding image was added to second sample
        tensor([[[[0.]]], [[[0.]]], [[[0.]]], [[[0.]]]])
    """
    # Text tokens can be handled independently by existing collater
    text_only = [
        {"tokens": sample["tokens"], "labels": sample["labels"]} for sample in batch
    ]
    collated_text = padded_collate(text_only, padding_idx, ignore_idx)
    max_seq_len = collated_text["tokens"].shape[-1]

    # TODO: Figure out how to make this more efficient or vectorized. Setting
    # max_num_tiles beforehand will save one nested for loop but may incur more
    # memory and compute costs in attention if max_num_tiles > batch_max_num_tiles

    # First loop: get max number of tiles in batch
    max_num_tiles = max(
        image.shape[0] for sample in batch for image in sample["images"]
    )
    # Second loop: pad images and masks to max number of tiles, max text seq len in batch
    batch_images = []
    batch_masks = []
    batch_aspect_ratios = []
    for sample in batch:
        sample_images = []
        sample_masks = []
        for image, mask in zip(sample["images"], sample["encoder_mask"]):
            # Single image in each sample has shape (n_tiles, c, h, w)
            n_tiles = image.shape[0]
            # Single mask in each sample corresponds to a single image and has shape (text_seq_len, image_seq_len)
            # where image_seq_len = n_tiles * tokens_per_tile
            text_seq_len, image_seq_len = mask.shape
            tokens_per_tile = image_seq_len // n_tiles
            padding_tiles = max_num_tiles - n_tiles
            padding_text = max_seq_len - text_seq_len
            # Image should now have shape (max_num_tiles, c, h, w)
            padded_image = F.pad(image, (0, 0, 0, 0, 0, 0, 0, padding_tiles), value=0)
            # Mask should now have shape (max_seq_len, max_image_seq_len), where
            # max_image_seq_len = max_num_tiles * tokens_per_tile
            padded_mask = F.pad(
                mask, (0, padding_tiles * tokens_per_tile, 0, padding_text), value=0
            )
            sample_images.append(padded_image)
            sample_masks.append(padded_mask)
        # Stack multiple images and masks per sample in num_images dimension
        batch_images.append(torch.stack(sample_images))
        batch_masks.append(torch.stack(sample_masks))
        batch_aspect_ratios.append(torch.stack(sample["aspect_ratio"]))
    # Finally, pad images, masks, aspect ratios to max number of images in batch
    # (bsz, max_num_images, max_num_tiles, c, h, w)
    collated_images = pad_sequence(batch_images, batch_first=True, padding_value=0)
    # (bsz, max_num_images, max_seq_len, max_image_seq_len)
    collated_masks = pad_sequence(batch_masks, batch_first=True, padding_value=0)
    # (bsz, max_num_images, 2)
    collated_aspect_ratios = pad_sequence(
        batch_aspect_ratios, batch_first=True, padding_value=1
    )

    return {
        "tokens": collated_text["tokens"],
        "labels": collated_text["labels"],
        "images": collated_images,
        "encoder_mask": collated_masks,
        "aspect_ratio": collated_aspect_ratios,
    }
