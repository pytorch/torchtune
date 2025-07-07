# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from torchtune.data import (
    CROSS_ENTROPY_IGNORE_IDX,
    left_pad_sequence,
    padded_collate_sft,
)


def qwen2_5_vl_padded_collate_images(
    batch: list[dict[str, Any]],
    padding_idx: int = 151655,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    pad_direction: str = "right",
    pad_to_multiple_of: int = 1,
) -> dict[str, torch.Tensor]:
    """
    Collate a batch of samples into a single dictionary.
    This is a modified version of padded_collate_tiled_images_and_mask that
    compresses images and grid_thw into single batch, due to encoder input
    signature.
    """

    if pad_direction not in ["left", "right"]:
        raise ValueError(
            f"pad_direction should be one of 'left' or 'right' but found {pad_direction}"
        )

    # Text tokens can be handled independently by existing collaters
    if pad_direction == "right":
        text_only = [
            {"tokens": sample["tokens"], "labels": sample["labels"]} for sample in batch
        ]
        collated_text = padded_collate_sft(
            text_only, padding_idx, ignore_idx, pad_to_multiple_of=pad_to_multiple_of
        )
    # For inference, we don't need to handle labels
    elif pad_direction == "left":
        if pad_to_multiple_of > 1:
            raise ValueError(
                f"pad_to_multiple_of={pad_to_multiple_of} is not supported for pad_direction='left'"
            )
        collated_text = {
            "tokens": left_pad_sequence(
                [torch.tensor(x["tokens"]) for x in batch],
                batch_first=True,
                padding_value=padding_idx,
            )
        }

    batch_dict = {
        "tokens": collated_text["tokens"],
    }
    if "labels" in collated_text:
        batch_dict["labels"] = collated_text["labels"]

    # compress images and grid_thw into single batch
    batch_images = []
    batch_grid_thw = []
    for sample in batch:
        sample_images = sample["encoder_input"]["image"]["hidden_states"]
        i, n, p = sample_images.shape
        sample_images = sample_images.reshape(i * n, p)

        # Stack multiple images per sample in num_images dimension
        batch_images.append(sample_images)
        batch_grid_thw.append(sample["encoder_input"]["image"]["grid_thw"])

    if "image" in batch[0]["encoder_input"]:
        batch_dict["encoder_input"] = {
            "image": {
                "hidden_states": torch.cat(batch_images),
                "grid_thw": torch.cat(batch_grid_thw),
            }
        }

    return batch_dict
