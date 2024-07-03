# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping, Protocol

from torchtune.data import Message
from torchtune.modules.tokenizers import ModelTokenizer


class Transform(Protocol):
    """
    Loose interface for all data and model transforms. Transforms operate at the
    sample level and perform operations on a sample dict which is contained in
    kwargs. Any fields that will be processed are unfolded with explicit keyword-arguments,
    then the updated dict is returned.
    """

    def __call__(self, **kwargs) -> Mapping[str, Any]:
        pass


class Compose(Transform):
    """
    Compose multiple transforms together, inspired by torchvision's ``Compose`` API

    Args:
        transforms (List[Transform]): List of transforms to compose together in sequential order.
    """

    def __init__(self, transforms: List[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, **kwargs) -> Mapping[str, Any]:
        for transform in self.transforms:
            kwargs = transform(**kwargs)
        return kwargs


class TokenizeMessages(Transform):
    """
    Apply the ``tokenize_messages`` method from a given
    :class:`~torchtune.modules.tokenizers.ModelTokenizer` on the ``messages`` field of the sample.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements
            the ``tokenize_messages`` method.
    """

    def __init__(self, tokenizer: ModelTokenizer, max_seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *, messages: List[Message], **kwargs) -> Mapping[str, Any]:
        tokens, mask = self.tokenizer.tokenize_messages(
            messages, max_seq_len=self.max_seq_len
        )
        kwargs.update({"tokens": tokens, "mask": mask})
        return kwargs


class CrossAttentionMask(Transform):
    """
    Computes the cross-attention mask for text + image inputs. Text tokens that
    participate in cross-attention with an image token will show True in the mask
    and follow these rules:
    1) Text tokens immediately following the image token up until the next image token
    2) Consecutive image tokens attend to all subsequent text tokens

    Resultant mask is of shape (text_seq_len, image_seq_len), where True indicates
    that the token outputted from the image encoder attends to the token in the
    text sequence.

    Args:
        num_patches (int): Number of patches per image, excluding class token.
        image_token_id (int): Token ID of the image special token.
    """

    def __init__(self, num_patches: int, image_token_id: int):
        self.num_patches = num_patches
        self.image_token_id = image_token_id

    def _get_image_attention_intervals(
        self, tokens: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Returns a list of tuples of the form (start, end) where start is the index
        of the current image token and end is the index of the next image token, exclusive.
        If the image token attends until the end of the sequence, end will be -1.
        """
        vision_token_locations = [
            i for i, token in enumerate(tokens) if token == self.image_token_id
        ]
        # Return empty list if there are no images
        if len(vision_token_locations) == 0:
            return []
        # If there is only one image, it will attend to subsequent text until end
        if len(vision_token_locations) == 1:
            return [[vision_token_locations[0], -1]]

        vision_masks = [
            [tok1, tok2]
            for tok1, tok2 in zip(
                vision_token_locations[:-1], vision_token_locations[1:]
            )
        ]
        # Last image will attend to subsequent text until end
        vision_masks.append([vision_token_locations[-1], -1])

        # If there are consecutive vision tokens, they should all attend to the
        # same subsequent text
        last_mask_end = vision_masks[-1][1]
        for vision_mask in vision_masks[::-1]:
            if vision_mask[0] == vision_mask[1] - 1:
                vision_mask[1] = last_mask_end
            last_mask_end = vision_mask[1]
        return vision_masks

    def __call__(self, *, tokens, images, **kwargs):
        # We are still at sample level pre-collating
        n_img, n_tiles, _, _, _ = images.shape
        text_seq_len = len(tokens)
        single_image_seq_len = n_tiles * self.num_patches + 1
        image_seq_len = single_image_seq_len * n_img
        intervals = self._get_image_attention_intervals(tokens)
        assert len(intervals) == n_img

        mask = torch.zeros(text_seq_len, image_seq_len, dtype=torch.bool)
        for image_num, interval in enumerate(intervals):
            start, end = interval
            end = text_seq_len if end == -1 else end
            mask[
                start:end,
                image_num
                * single_image_seq_len : (image_num + 1)
                * single_image_seq_len,
            ] = True

        kwargs.update({"encoder_mask": mask, "tokens": tokens, "images": images})
        return kwargs
