# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX, PACK_TYPE
from torchtune.modules.attention_utils import packed_block_causal_mask


def left_pad_sequence(
    sequences: List[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0,
) -> torch.Tensor:
    """
    This function is identical to :func:`torch.nn.utils.rnn.pad_sequence`, but
    instead pads a list of variable length Tensors from the left to the length
    of the longest sequence.

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (List[torch.Tensor]): list of variable length sequences.
        batch_first (bool): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise. Default False.
        padding_value (float): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise

    Example:
        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([4, 5, 6, 7])
        >>> c = torch.tensor([8, 9, 10, 11, 12])
        >>> left_pad_sequence([a, b, c], batch_first=True, padding_value=0)
        tensor([[ 0,  0,  1,  2,  3],
                [ 0,  4,  5,  6,  7],
                [ 8,  9, 10, 11, 12]])
    """
    return pad_sequence(
        map(lambda x: torch.flip(x, dims=[0]), sequences),
        batch_first=batch_first,
        padding_value=padding_value,
    ).flip(dims=[int(batch_first)])


def padded_collate(
    batch: List[Dict[str, List[int]]],
    *,
    pad_direction: str,
    keys_to_pad: List[str],
    padding_idx: Union[int, Dict[str, int]],
):
    """
    A generic padding collation function which pads ``keys_to_pad`` entries in a
    batch of sequences from the given ``pad_direction`` to the maximum sequence length for
    each entry in the batch.

    Note:
        This function assumes all batch elements which are not in ``keys_to_pad`` do not require
        any collation (see example below).

    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries containing inputs.
        pad_direction (str): whether to pad entries from the left, or right. If ``pad_direction="right"``, we use
            :func:`torch.nn.utils.rnn.pad_sequence`, otherwise if ``pad_direction="left"``,
            we use :func:`torchtune.data.left_pad_sequence`.
        keys_to_pad (List[str]): Batch element keys to apply padding to. Should be a subset
            of keys in the batch.
        padding_idx (Union[int, Dict[str, int]]): Either a single integer padding value to apply to all
            ``keys_to_pad`` elements, or a mapping with keys identical to ``keys_to_pad`` with per-key
            padding values.

    Returns:
        torch.Tensor: The padded tensor of input ids with shape [batch_size, max_seq_len].

    Raises:
        ValueError: if ``pad_direction`` is not one of "left" or "right".
        ValueError: if ``keys_to_pad`` is empty, or is not a list, or is not a subset of keys in the batch.
        ValueError: if ``padding_idx`` is provided as a dictionary, but the keys are not identical to
            ``keys_to_pad``.

    Example:
        >>> a = [1, 2, 3]
        >>> b = [4, 5, 6, 7]
        >>> c = [8, 9, 10, 11, 12]
        >>> batch = [
        >>>     {"tokens": a, "labels": 1},
        >>>     {"tokens": b, "labels": 3},
        >>>     {"tokens": c, "labels": 0},
        >>> ]
        >>> padded_collate(
        >>>     batch,
        >>>     pad_direction="left",
        >>>     keys_to_pad=["tokens"],
        >>>     padding_idx=-10
        >>> )
        {
            'labels': tensor([1, 3, 0]),
            'tokens': tensor([[-10, -10,   1,   2,   3],
                              [-10,   4,   5,   6,   7],
                              [  8,   9,  10,  11,  12]])
        }
    """
    if pad_direction not in ["left", "right"]:
        raise ValueError(
            f"pad_direction should be one of 'left' or 'right' but found {pad_direction}"
        )

    if not isinstance(keys_to_pad, list) or not keys_to_pad:
        raise ValueError(
            f"keys_to_pad should be a list of strings with at least one element, but found {keys_to_pad}!"
        )

    keys_to_pad = set(keys_to_pad)
    if isinstance(padding_idx, dict):
        if not set(padding_idx.keys()) == keys_to_pad:
            raise ValueError(
                f"padding_idx was provided as a dictionary, but the keys ({padding_idx.keys()}) "
                f"are not the same as keys_to_pad ({keys_to_pad})"
            )
        if not keys_to_pad <= set(batch[0].keys()):
            raise ValueError(
                "keys_to_pad should be a subset of keys in the batch, but found "
                f"{keys_to_pad} and {set(batch[0].keys())}, respectively."
            )

    # let's pull out any batch elements which don't need any padding
    # and convert to tensors
    batch_keys = [k for k in batch[0].keys() if k not in keys_to_pad]
    output_dict = {k: torch.tensor([x[k] for x in batch]) for k in batch_keys}

    # now pad the remaining keys
    pad_fn = (
        torch.nn.utils.rnn.pad_sequence
        if pad_direction == "right"
        else left_pad_sequence
    )
    for k in keys_to_pad:
        output_dict[k] = pad_fn(
            [torch.tensor(x[k]) for x in batch],
            batch_first=True,
            padding_value=padding_idx[k]
            if isinstance(padding_idx, dict)
            else padding_idx,
        )
    return output_dict


def padded_collate_sft(
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


# TODO: Generalize this to support any type of encoder input, right now this assumes
# a specific encoder_input signature
def padded_collate_tiled_images_and_mask(
    batch: List[Dict[str, Any]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    pad_direction: str = "right",
    pad_max_tiles: Optional[int] = None,
    pad_max_images: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Pad a batch of text sequences, tiled image tensors, aspect ratios,
    and cross attention masks. This can be used for both training and inference.

    ``batch`` is expected to be a list of sample dicts containing the following::
        - "tokens": List[int] of length text_seq_len, varies across samples
        - "labels": List[int] of length text_seq_len, varies across samples
        - "encoder_input": Dict[str, List[torch.Tensor]]
            - "images": List[torch.Tensor], each with shape (n_tiles, c, h, w)
            - "aspect_ratio": List[torch.Tensor], each with shape (2, ) to indicate h_ratio, w_ratio
        - "encoder_mask": List[Tensor], each with shape (text_seq_len, image_seq_len)

    Shape notation:
        - c = channel dim
        - h = height dim
        - w = weight dim

    Note:
        For each element in the batch, ``len(images) == len(encoder_mask) == len(aspect_ratio)``.

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
        pad_direction (str): whether to pad entries from the left, or right. If ``pad_direction="right"``, we use
            :func:`torch.nn.utils.rnn.pad_sequence`, otherwise if ``pad_direction="left"``,
            we use :func:`torchtune.data.left_pad_sequence`. For training, we typically want to pad from the right.
            For inference, we typically want to pad from the left. Defaults to "right".
        pad_max_tiles (Optional[int]): Maximum number of tiles to pad to. If None, will pad to the largest number of tiles
            in the batch. Defaults to None.
        pad_max_images (Optional[int]): Maximum number of images to pad to. If None, will pad to the largest number of images
            in the batch. Defaults to None.

    Returns:
        Dict[str, Tensor]: Collated tokens, labels, images, encoder_mask, aspect_ratio tensors.
            - tokens: Tensor of shape (bsz, max_seq_len)
            - labels: Tensor of shape (bsz, max_seq_len)
            - images: Tensor of shape (bsz, max_num_images, max_num_tiles, c, h, w)
            - encoder_mask: Tensor of shape (bsz, max_seq_len, tokens_per_tile * max_num_tiles * max_num_images)
            - aspect_ratio: Tensor of shape (bsz, max_num_images, 2)

    Raises:
        ValueError: if ``pad_direction`` is not one of "left" or "right".
        ValueError: if pad_max_tiles is set to a value less than the largest number of tiles in an image.

    Example:
        >>> image_id = 1
        >>> tokens_per_tile = 5
        >>> c, h, w = 1, 1, 1
        >>> batch = [
        ...     {
        ...         "tokens": [1, 2, 1, 3], "labels": [4, 5, 6, 7],
        ...         "encoder_input": {
        ...             # One image with two tiles, one image with three tiles
        ...             "images": [torch.ones(2, c, h, w), torch.ones(3, c, h, w)],
        ...             "aspect_ratio": [torch.tensor([1, 2]), torch.tensor([1, 3])],
        ...         },
        ...         # Mask is shape (text_seq_len, tokens_per_tile * n_tiles)
        ...         "encoder_mask": [torch.ones(4, 5 * 2), torch.ones(4, 5 * 3)],
        ...     },
        ...     {
        ...         "tokens": [1, 4], "labels": [8, 9],
        ...         "encoder_input": {
        ...             # One image with four tiles
        ...             "images": [torch.ones(4, c, h, w)],
        ...             "aspect_ratio": [torch.tensor([2, 2])],
        ...         },
        ...         # Mask is shape (text_seq_len, tokens_per_tile * n_tiles)
        ...         "encoder_mask": [torch.ones(2, 5 * 4)],
        ...     },
        ... ]
        >>> model_inputs = padded_collate_tiled_images_and_mask(batch=batch)
        >>> print(model_inputs["tokens"])
        tensor([[1, 2, 1, 3],
                [1, 4, 0, 0]])
        >>> print(model_inputs["labels"])
        tensor([[4, 5, 6, 7],
                [8, 9, -100, -100]])
        >>> print(model_inputs["encoder_input"]["images"].shape)  # (bsz, max_num_images, max_num_tiles, c, h, w)
        torch.Size([2, 2, 4, 1, 1, 1])
        >>> print(model_inputs["encoder_mask"].shape)  # (bsz, max_text_seq_len, tokens_per_tile * max_num_tiles * max_num_images)
        torch.Size([2, 4, 40])
        >>> print(model_inputs["encoder_input"]["aspect_ratio"].shape)  # (bsz, max_num_images, 2)
        torch.Size([2, 2, 2])
        >>> print(model_inputs["encoder_input"]["images"][0, 0, ...])  # Image with two tiles got padded to four
        tensor([[[[1.]]], [[[1.]]], [[[0.]]], [[[0.]]]])
        >>> print(model_inputs["encoder_input"]["images"][0, 1, ...])  # Image with three tiles got padded to four
        tensor([[[[1.]]], [[[1.]]], [[[1.]]], [[[0.]]]])
        >>> print(model_inputs["encoder_input"]["images"][1, 0, ...])  # Image with four tiles did not get padded
        tensor([[[[1.]]], [[[1.]]], [[[1.]]], [[[1.]]]])
        >>> print(model_inputs["encoder_input"]["images"][1, 1, ...])  # Extra padding image was added to second sample
        tensor([[[[0.]]], [[[0.]]], [[[0.]]], [[[0.]]]])
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
        collated_text = padded_collate_sft(text_only, padding_idx, ignore_idx)
    # For inference, we don't need to handle labels
    elif pad_direction == "left":
        collated_text = {
            "tokens": left_pad_sequence(
                [torch.tensor(x["tokens"]) for x in batch],
                batch_first=True,
                padding_value=padding_idx,
            )
        }

    max_seq_len = collated_text["tokens"].shape[-1]
    bsz = len(batch)

    # TODO: Figure out how to make this more efficient or vectorized. Setting
    # max_num_tiles beforehand will save one nested for loop but may incur more
    # memory and compute costs in attention if max_num_tiles > batch_max_num_tiles

    # First loop: get max number of tiles in batch
    max_num_tiles = max(
        image.shape[0]
        for sample in batch
        for image in sample["encoder_input"]["images"]
    )
    if pad_max_tiles is not None:
        if pad_max_tiles < max_num_tiles:
            raise ValueError(
                f"More tiles in image {max_num_tiles}, than pad_max_tiles {pad_max_tiles}"
            )
        max_num_tiles = pad_max_tiles

    # Second loop: pad images and masks to max number of tiles, max text seq len in batch
    batch_images = []
    batch_masks = []
    batch_aspect_ratios = []
    for sample in batch:
        sample_images = []
        sample_masks = []
        for image, mask in zip(
            sample["encoder_input"]["images"], sample["encoder_mask"]
        ):
            # Single image in each sample has shape (n_tiles, c, h, w)
            n_tiles = image.shape[0]
            # Single mask in each sample corresponds to a single image and has shape (text_seq_len, image_seq_len)
            # where image_seq_len = n_tiles * tokens_per_tile
            text_seq_len, image_seq_len = mask.shape
            tokens_per_tile = image_seq_len // n_tiles
            padding_tiles = max_num_tiles - n_tiles
            right_padding_text = (
                max_seq_len - text_seq_len if pad_direction == "right" else 0
            )
            left_padding_text = (
                max_seq_len - text_seq_len if pad_direction == "left" else 0
            )

            # Image should now have shape (max_num_tiles, c, h, w)
            padded_image = F.pad(image, (0, 0, 0, 0, 0, 0, 0, padding_tiles), value=0)
            # Mask should now have shape (max_seq_len, max_image_seq_len), where
            # max_image_seq_len = max_num_tiles * tokens_per_tile
            padded_mask = F.pad(
                mask,
                (
                    0,
                    padding_tiles * tokens_per_tile,
                    left_padding_text,
                    right_padding_text,
                ),
                value=0,
            )

            sample_images.append(padded_image)
            sample_masks.append(padded_mask)
        # Stack multiple images and masks per sample in num_images dimension
        batch_images.append(torch.stack(sample_images))
        batch_masks.append(torch.stack(sample_masks))
        batch_aspect_ratios.append(torch.stack(sample["encoder_input"]["aspect_ratio"]))
    # Finally, pad images, masks, aspect ratios to max number of images in batch
    # (bsz, max_num_images, max_num_tiles, c, h, w)
    collated_images = pad_sequence(batch_images, batch_first=True, padding_value=0)
    # (bsz, max_num_images, max_seq_len, max_image_seq_len)
    collated_masks = pad_sequence(batch_masks, batch_first=True, padding_value=0)
    # (bsz, max_num_images, 2)
    collated_aspect_ratios = pad_sequence(
        batch_aspect_ratios, batch_first=True, padding_value=1
    )

    # Concatenate masks for multiple images across image_seq_len dimension
    concat_masks = collated_masks.view(bsz, max_seq_len, -1)
    if pad_max_images is not None:
        _, _, img_seq = concat_masks.shape
        concat_masks = F.pad(
            concat_masks, (0, pad_max_images * image_seq_len - img_seq)
        )

    batch_dict = {
        "tokens": collated_text["tokens"],
        "encoder_input": {
            "images": collated_images,
            "aspect_ratio": collated_aspect_ratios,
        },
        "encoder_mask": concat_masks,
    }

    if "labels" in collated_text:
        batch_dict["labels"] = collated_text["labels"]

    return batch_dict


def padded_collate_packed(
    batch: List[PACK_TYPE],
) -> Dict[str, torch.Tensor]:
    """Collate packed sequences into a batch. Only convert the seq lens into
    a block mask for use with attention. Tokens, labels, and input_pos are
    already padded to the same length within :class:`~torchtune.datasets.PackedDataset`.

    Args:
        batch (List[PACK_TYPE]): A list of pack dictionaries containing the following keys:
            - tokens: input token ids
            - labels: label token ids
            - input_pos: relative position ids for each sequence in pack
            - seq_lens: lengths of each sample within the pack

    Returns:
        Dict[str, torch.Tensor]: Collated input, label, input_pos, mask tensors.

    Example:
        >>> token_pairs = [
        >>>    {"tokens": [1, 2, 3, 4, 5, 6], "labels": [7, 8, 9, 10, 11, 12],
        >>>     "input_pos": [0, 1, 2, 0, 1, 0], "seq_lens": [3, 2, 1]},
        >>>    {"tokens": [13, 14, 15, 16, 17, 18], "labels": [19, 20, 21, 22, 23, 24],
        >>>     "input_pos": [0, 1, 0, 1, 0, 1], "seq_lens": [2, 2, 2]},
        >>> ]
        >>> collated = padded_collate_packed(
        >>>    batch=token_pairs,
        >>>    device=device,
        >>> )
        >>> collated["mask"]
        >>> tensor([
        >>> [[1, 0, 0, 0, 0, 0],
        >>>  [1, 1, 0, 0, 0, 0],
        >>>  [1, 1, 1, 0, 0, 0],
        >>>  [0, 0, 0, 1, 0, 0],
        >>>  [0, 0, 0, 1, 1, 0],
        >>>  [0, 0, 0, 0, 0, 1]],
        >>> [[1, 0, 0, 0, 0, 0],
        >>>  [1, 1, 0, 0, 0, 0],
        >>>  [0, 0, 1, 0, 0, 0],
        >>>  [0, 0, 1, 1, 0, 0],
        >>>  [0, 0, 0, 0, 1, 0],
        >>>  [0, 0, 0, 0, 1, 1]])
    """

    tokens = torch.stack([x["tokens"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    input_pos = torch.stack([x["input_pos"] for x in batch])
    seq_lens = [x["seq_lens"] for x in batch]

    block_mask = packed_block_causal_mask(
        seq_lens=seq_lens,
    )

    return {
        "tokens": tokens,
        "labels": labels,
        "input_pos": input_pos,
        "mask": block_mask,
    }


def padded_collate_dpo(
    batch: List[Dict[str, List[int]]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of sequences for Direct Preference Optimization (DPO).

    This function takes a batch of sequences, where each sequence is represented
    as a dictionary with multiple key-value pairs. Each key corresponds to a different
    sequence component, such as input_ids or labels.

    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries, where each dictionary
            represents a sequence with multiple components, 'chosen_input_ids',
            'chosen_labels', 'rejected_input_ids', and 'rejected_labels' are required.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing concatenated and padded
        input ids and labels.

    Example:
        >>> batch = [
        >>>    {'chosen_input_ids': [1, 2, 3], 'rejected_input_ids': [4, 5],
        >>>      'chosen_labels': [6, 7, 8], 'rejected_labels': [9, 10]},
        >>>    {'chosen_input_ids': [11, 12], 'rejected_input_ids': [13, 14, 15],
        >>>      'chosen_labels': [16, 17], 'rejected_labels': [18, 19, 20]},
        >>> ]
        >>> padded_collate_dpo(batch)
        >>> (tensor([[ 1,  2,  3],
        >>>          [11, 12,  0],
        >>>          [ 4,  5,  0],
        >>>          [13, 14, 15]]),
        >>>  tensor([[ 6,  7,  8],
        >>>          [16, 17, -100],
        >>>          [ 9, 10, -100],
        >>>          [18, 19, 20]]))
    """
    chosen_input_ids = [torch.tensor(ex["chosen_input_ids"]) for ex in batch]
    rejected_input_ids = [torch.tensor(ex["rejected_input_ids"]) for ex in batch]
    chosen_labels = [torch.tensor(ex["chosen_labels"]) for ex in batch]
    rejected_labels = [torch.tensor(ex["rejected_labels"]) for ex in batch]

    to_pad_input_ids = chosen_input_ids + rejected_input_ids
    to_pad_labels = chosen_labels + rejected_labels

    concatenated_input_ids = pad_sequence(
        to_pad_input_ids, batch_first=True, padding_value=padding_idx
    )
    concatenated_labels = pad_sequence(
        to_pad_labels, batch_first=True, padding_value=ignore_idx
    )

    return concatenated_input_ids, concatenated_labels
