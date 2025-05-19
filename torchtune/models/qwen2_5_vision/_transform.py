# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torchvision.transforms.v2 as v2
from PIL import Image

from torchtune.data.message import Message
from torchtune.data.templates import _TemplateType, _get_prompt_template
from torchtune.models.clip._transform import CLIPImageTransform
from torchtune.models.qwen2_5._tokenizer import Qwen2_5Tokenizer
from torchtune.tokenizers.utils import parse_hf_tokenizer_json

logger = logging.getLogger(__name__)


class Qwen25VisionTransform:
    """
    Transform for Qwen 2.5 Vision model that handles both text tokenization and image processing.

    Args:
        path (str): Path to the tokenizer model file.
        tile_size (int): Size of the image tiles.
        patch_size (int): Size of the patches within each tile.
        max_num_tiles (int): Only used if possible_resolutions is NOT given.
            Maximum number of tiles to break an image into.
            This will be used to generate possible_resolutions,
            e.g. [(224, 224), (224, 448), (448, 224)] if ``max_num_tiles = 2`` and ``tile_size = 224``.
            Default 4.
        pixel_shuffle_scaling_factor (float): scaling factor for pixel shuffle. Default is 0.5. You must ensure this
            matches the pixel shuffle scaling factor used in the vision projection head if modified from default.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Qwen 2.5 special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        image_mean (Optional[List[float]]): Mean values of each channel, used for normalization.
        image_std (Optional[List[float]]): Standard deviations for each channel, used for normalization.
        dtype (torch.dtype): Data type of transformed image. Default torch.bfloat16.
        prompt_template (Optional[_TemplateType]): template used to format the messages based on their role.

    Examples:
        >>> model_transform = Qwen25VisionTransform("/path/to/tokenizer.model", tile_size=224, patch_size=14)
        >>> transformed_data = model_transform({"messages": user_message, "images": [img1, img2]})
        >>> print(transformed_data["tokens"])
        [1, 31587, 29644, 102, 2]
        >>> print(transformed_data["images"][0].shape)
        torch.Size([4, 3, 224, 224])
    """

    def __init__(
        self,
        path: str,
        *,
        tile_size: int,
        patch_size: int,
        max_num_tiles: int = 4,
        pixel_shuffle_scaling_factor: float = 0.5,
        special_tokens_path: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        dtype: torch.dtype = torch.bfloat16,
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
        self.tokenizer = Qwen2_5Tokenizer(
            path=path,
            special_tokens=special_tokens,
            max_seq_len=max_seq_len,
            prompt_template=template,
        )
        self.thumbnail_transform = v2.Compose(
            [
                v2.Resize((tile_size, tile_size)),
                v2.ToImage(),
                v2.ToDtype(dtype=dtype, scale=True),
                v2.Normalize(mean=image_mean, std=image_std, inplace=True),
            ]
        )
        self.clip_transform = CLIPImageTransform(
            image_mean=image_mean,
            image_std=image_std,
            tile_size=tile_size,
            possible_resolutions=None,
            max_num_tiles=max_num_tiles,
            resample="bilinear",
            resize_to_max_canvas=False,
            dtype=dtype,
        )

        self.stop_tokens = self.tokenizer.stop_tokens
        self.special_tokens = self.tokenizer.special_tokens
        self.max_seq_len = max_seq_len
        self.max_num_tiles = max_num_tiles
        patch_grid_size = tile_size // patch_size
        self.patches_per_tile = patch_grid_size**2
        self.image_seq_len = max_num_tiles * self.patches_per_tile  # No CLS token
        self.pixel_shuffle_scaling_factor = pixel_shuffle_scaling_factor
        # Number of patches in each tile in image tensor after accounting for pixel shuffling.
        self.patch_tokens_per_tile = int(
            self.patches_per_tile * (self.pixel_shuffle_scaling_factor**2)
        )
        self.prompt_template = prompt_template
        self.pad_id = self.tokenizer.pad_id

    @property
    def base_vocab_size(self) -> int:
        return self.tokenizer.base_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def transform_image(
        self, image: Image.Image, inference: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform an image into tiles for the vision encoder.

        Args:
            image (Image.Image): The input image.
            inference (bool): Whether to run in inference mode. Default is False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The transformed image tiles and aspect ratio.
        """
        if inference:
            # For inference, we use the thumbnail transform
            image_tensor = self.thumbnail_transform(image)
            return image_tensor.unsqueeze(0), torch.tensor([1, 1])
        else:
            # For training, we use the CLIP transform
            sample = {"image": image}
            transformed = self.clip_transform(sample)
            return transformed["image"], transformed["aspect_ratio"]

    def tokenize_message(
        self,
        message: Message,
        *,
        add_start_tokens: bool = True,
        add_end_tokens: bool = True,
    ) -> List[int]:
        """
        Tokenize a single message into a list of token ids.

        Args:
            message (Message): The message to tokenize.
            add_start_tokens (bool): Whether to add the tokenizer's bos_id. Default True.
            add_end_tokens (bool): Whether to add the tokenizer's eos_id. Default True.

        Returns:
            List[int]: The list of token ids.
        """
        return self.tokenizer.tokenize_message(
            message=message,
            add_start_tokens=add_start_tokens,
            add_end_tokens=add_end_tokens,
        )

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_end_tokens: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Tokenize a list of messages into a list of token ids and masks.

        Args:
            messages (List[Message]): The list of messages to tokenize.
            add_end_tokens (bool): Whether to add the tokenizer's eos_id. Default True.

        Returns:
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.
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
                - tokens: List[int] of tokenized messages
                - mask: List[bool] of masks for the tokenized messages
                - encoder_input: Dict[str, Any] of transformed images
        """
        encoder_input = {"vision": {"images": []}}
        messages = sample["messages"]
        for message in messages:
            for content in message.content:
                if content["type"] == "image":
                    image = content["content"]
                    tiles, ar = self.transform_image(image, inference=inference)
                    encoder_input["vision"]["images"].append(tiles)

                    # Add number of patch tokens, tiles, and aspect ratio to metadata
                    # so tokenizer can add the corresponding special tokens
                    content["patch_tokens_per_tile"] = self.patch_tokens_per_tile
                    content["aspect_ratio"] = ar

        sample["encoder_input"] = encoder_input
        sample = self.tokenizer(sample, inference=inference)
        return sample
