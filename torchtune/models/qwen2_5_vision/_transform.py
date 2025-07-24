# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Any, Mapping, Optional

import torch
from PIL import Image

from torchtune.data import Message
from torchtune.data._prompt_templates import _get_prompt_template, _TemplateType
from torchtune.models.qwen2_5_vision._tokenizer import Qwen25VLTokenizer
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import (
    ModelTokenizer,
    parse_hf_tokenizer_json,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F

logger = logging.getLogger(__name__)

# HuggingFace OPENAI_CLIP constants to match their normalization
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 12845056,
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen25VLImageTransform:
    """
    This class accepts images of any size and dynamically resizes, normalizes and patches it
    based on the image size constraints and patch size.

    Args:
        image_mean (Optional[list[float]]): Mean values of each channel, used for normalization.
            Should be the same used for the pre-trained model. If None, uses OPENAI_CLIP_MEAN. Default None.
        image_std (Optional[list[float]]): Standard deviation values of each channel, used for normalization.
            Should be the same used for the pre-trained model. If None, uses OPENAI_CLIP_STD. Default None.
        patch_size (int): Size of the patches to divide the image into. Default 14.
        merge_size (int): Size of the patch merging factor. Default 2.
        temporal_patch_size (int): Size of the temporal patch merging factor. Default 2.
        size (Optional[dict[str, int]]): Size configuration with 'shortest_edge' and 'longest_edge' keys.
        min_pixels (Optional[int]): Minimum number of pixels for the shorter edge. Default 3136 (56 * 56).
        max_pixels (Optional[int]): Maximum number of pixels for the longer edge. Default 1003520 (28 * 28 * 1280).
        dtype (torch.dtype): Data type of the output image. Default torch.float32.
        resample (str): Resampling method used when resizing images. Supports any enum of
            ``torchvision.transforms.InterpolationMode``, e.g. "nearest", "nearest_exact", "bilinear", "bicubic".
            Default 'bicubic'.

    Raises:
        ValueError: If size is provided but does not contain 'shortest_edge' and 'longest_edge' keys.
    """

    def __init__(
        self,
        *,
        image_mean: Optional[list[float]] = None,
        image_std: Optional[list[float]] = None,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        size: Optional[dict[str, int]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        resample: str = "bicubic",
    ) -> None:
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size

        # Handle size configuration - prioritize size dict over individual params
        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError(
                    "size must contain 'shortest_edge' and 'longest_edge' keys."
                )
            self.size = size.copy()
        else:
            self.size = {"shortest_edge": 56 * 56, "longest_edge": 12845056}

        # Override with individual parameters if provided
        if min_pixels is not None:
            self.size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            self.size["longest_edge"] = max_pixels

        self.min_pixels = self.size["shortest_edge"]
        self.max_pixels = self.size["longest_edge"]

        self.dtype = dtype
        self.resample = getattr(InterpolationMode, resample.upper())

        # Use OPENAI_CLIP defaults if not provided (matches HuggingFace)
        self.mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.std = image_std if image_std is not None else OPENAI_CLIP_STD

    def __call__(
        self, sample: Mapping[str, Any], inference: bool = False
    ) -> Mapping[str, Any]:
        """
        Apply image decoding and transformations to the "image" field in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with an "image" field containing
                a PIL Image or torch.Tensor
            inference (bool): Whether the template is being used for inference or not.

        Returns:
            Mapping[str, Any]: The sample with updated fields:
                - "pixel_values": Flattened patches tensor
                - "image_grid_thw": Grid dimensions (temporal, height, width)
                - "num_patches": Number of patches calculated
        """
        image = sample["image"]
        assert isinstance(
            image, (Image.Image, torch.Tensor)
        ), "Input image must be a PIL image or a torch.Tensor."

        # Convert to RGB and tensor
        if isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")
        image = F.to_image(image)

        # Convert to float and rescale to [0, 1] - this matches HF's rescaling step
        image = F.to_dtype(image, dtype=torch.float32, scale=True)

        # Get image dimensions
        height, width = image.shape[-2:]

        # Calculate resize dimensions
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        # Resize image
        image = F.resize(
            image, size=(resized_height, resized_width), interpolation=self.resample
        )

        # Normalize with OPENAI_CLIP values
        image = F.normalize(image, mean=self.mean, std=self.std)

        image = image.to(dtype=self.dtype)

        patches = image.unsqueeze(0)

        if patches.shape[0] % self.temporal_patch_size != 0:
            repeats_needed = self.temporal_patch_size - (
                patches.shape[0] % self.temporal_patch_size
            )
            last_frame = patches[-1:].repeat(repeats_needed, 1, 1, 1)
            patches = torch.cat([patches, last_frame], dim=0)

        # Calculate grid dimensions
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )
        channels = patches.shape[1]

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channels,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )

        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)

        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channels * self.temporal_patch_size * self.patch_size * self.patch_size,
        )

        num_patches = grid_h * grid_w
        num_image_tokens = num_patches // self.merge_size**2

        sample.update(
            {
                "pixel_values": flatten_patches,
                "image_grid_thw": torch.tensor([[grid_t, grid_h, grid_w]]),
                "num_image_tokens": num_image_tokens,
            }
        )

        return sample


class Qwen25VLTransform(ModelTokenizer, Transform):
    """
    Transform for Qwen 2.5 Vision model that handles both text tokenization and image processing.

    Args:
        path (str): Path to the tokenizer vocab.json file.
        merges_file (str): Path to the tokenizer merges.txt file.
        patch_size (int): Size of the patches used in vision processing. Default 14.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Qwen 2.5 special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        image_mean (Optional[list[float]]): Mean values of each channel, used for normalization.
            Default None to use OPENAI_CLIP_MEAN.
        image_std (Optional[list[float]]): Standard deviations for each channel, used for normalization.
            Default None to use OPENAI_CLIP_STD.
        dtype (torch.dtype): Data type of transformed image. Default torch.float32.
        prompt_template (Optional[_TemplateType]): template used to format the messages based on their role.
    """

    def __init__(
        self,
        path: str,
        merges_file: str,
        *,
        patch_size: int = 14,
        special_tokens_path: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        image_mean: Optional[list[float]] = None,
        image_std: Optional[list[float]] = None,
        dtype: torch.dtype = torch.float32,
        prompt_template: Optional[_TemplateType] = None,
    ):
        special_tokens = (
            parse_hf_tokenizer_json(special_tokens_path)
            if special_tokens_path is not None
            else None
        )
        template = (
            _get_prompt_template(prompt_template)
            if prompt_template is not None
            else None
        )
        self.tokenizer = Qwen25VLTokenizer(
            path=path,
            merges_file=merges_file,
            max_seq_len=max_seq_len,
            prompt_template=template,
        )

        # Initialize the Qwen2.5 VL image transform
        self.image_transform = Qwen25VLImageTransform(
            image_mean=image_mean,
            image_std=image_std,
            patch_size=patch_size,
            merge_size=2,  # Default merge size for Qwen2.5-VL
            temporal_patch_size=2,  # Default temporal patch size
            dtype=dtype,
            resample="bicubic",
        )

        self.stop_tokens = self.tokenizer.stop_tokens
        self.special_tokens = self.tokenizer.special_tokens
        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        self.prompt_template = prompt_template
        self.pad_id = self.tokenizer.pad_id

    @property
    def base_vocab_size(self) -> int:
        return len(self.tokenizer.encoder)

    @property
    def vocab_size(self) -> int:
        # Total vocab size includes base vocab + special tokens
        return len(self.tokenizer.encoder) + len(self.tokenizer.special_tokens)

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """
        Encode a string into a list of token ids.

        Args:
            text (str): The string to encode.
            add_bos (bool): Whether to add the tokenizer's bos_id. Default is True.
            add_eos (bool): Whether to add the tokenizer's eos_id. Default is True.

        Returns:
            list[int]: The list of token ids.
        """
        return self.tokenizer.encode(text=text, add_bos=add_bos, add_eos=add_eos)

    def decode(
        self,
        token_ids: list[int],
        truncate_at_eos: bool = True,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode a list of token ids into a string.

        Args:
            token_ids (list[int]): The list of token ids.
            truncate_at_eos (bool): Whether to truncate the string at the end of
                sequence token. Default is True.
            skip_special_tokens (bool): Whether to show or skip special tokens in the decoded string.
                Default is True.

        Returns:
            str: The decoded string.
        """
        if truncate_at_eos and self.tokenizer.eos_id in token_ids:
            eos_index = token_ids.index(self.tokenizer.eos_id)
            token_ids = token_ids[:eos_index]

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def transform_image(
        self, image: Image.Image, inference: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Transform an image into flattened patches for the vision encoder.
        This method applies the transformations defined in `Qwen25VLImageTransform`.

        Args:
            image (Image.Image): The input image.
            inference (bool): Whether to run in inference mode. This is passed to the
                underlying image transform. Default is False.

        Returns:
            tuple[torch.Tensor, torch.Tensor, int]: A tuple containing:
                - The transformed image patches as a tensor.
                - The image grid dimensions (t, h, w) as a tensor.
                - The number of patches calculated.
        """
        sample = {"image": image}
        transformed = self.image_transform(sample, inference=inference)
        return (
            transformed["pixel_values"],
            transformed["image_grid_thw"],
            transformed["num_image_tokens"],
        )

    def tokenize_message(
        self,
        message: Message,
        *,
        add_start_tokens: bool = True,
        add_end_tokens: bool = True,
    ) -> list[int]:
        """
        Tokenize a single message into a list of token ids.

        Args:
            message (Message): The message to tokenize.
            add_start_tokens (bool): Whether to add the tokenizer's bos_id. Default True.
            add_end_tokens (bool): Whether to add the tokenizer's eos_id. Default True.

        Returns:
            list[int]: The list of token ids.
        """
        return self.tokenizer.tokenize_message(
            message=message,
            add_start_tokens=add_start_tokens,
            add_end_tokens=add_end_tokens,
        )

    def tokenize_messages(
        self,
        messages: list[Message],
        *,
        add_end_tokens: bool = True,
    ) -> tuple[list[int], list[bool]]:
        """
        Tokenize a list of messages into a list of token ids and masks.

        Args:
            messages (list[Message]): The list of messages to tokenize.
            add_end_tokens (bool): Whether to add the tokenizer's eos_id. Default True.

        Returns:
            tuple[list[int], list[bool]]: The list of token ids and the list of masks.
        """
        return self.tokenizer.tokenize_messages(
            messages=messages,
            add_end_tokens=add_end_tokens,
        )

    def __call__(
        self, sample: Mapping[str, Any], inference: bool = False
    ) -> Mapping[str, Any]:
        """
        Apply image decoding, transformations and tokenization to messages in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with a "messages" field.
            inference (bool): Whether to run in inference mode. Default is False.

        Returns:
            Mapping[str, Any]: The transformed sample with the following fields:
                - tokens: list[int] of tokenized messages
                - mask: list[bool] of masks for the tokenized messages
                - encoder_input: dict[str, Any] of transformed images
        """
        encoder_input = {"image": {"hidden_states": [], "grid_thw": []}}
        messages = sample["messages"]
        for message in messages:
            for content in message.content:
                if content["type"] == "image":
                    image = content["content"]

                    (
                        pixel_values,
                        image_grid_thw,
                        num_image_tokens,
                    ) = self.transform_image(image, inference=inference)

                    content["num_image_tokens"] = num_image_tokens

                    encoder_input["image"]["hidden_states"].append(pixel_values)
                    encoder_input["image"]["grid_thw"].append(image_grid_thw)

        encoder_input["image"]["hidden_states"] = torch.stack(
            encoder_input["image"]["hidden_states"], dim=0
        )
        encoder_input["image"]["grid_thw"] = torch.cat(
            encoder_input["image"]["grid_thw"], dim=0
        )

        sample["encoder_input"] = encoder_input
        sample = self.tokenizer(sample, inference=inference)
        return sample
