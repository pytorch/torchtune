# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping, Optional, Tuple

import torch
from PIL import Image

from torchtune.data import Message
from torchtune.data._prompt_templates import _get_prompt_template, _TemplateType

from torchtune.models.clip import CLIPImageTransform
from torchtune.models.llama4._tokenizer import Llama4Tokenizer
from torchtune.modules.tokenizers import ModelTokenizer, parse_hf_tokenizer_json
from torchtune.modules.transforms import Transform
from torchvision.transforms import v2


class Llama4Transform(ModelTokenizer, Transform):
    """
    This transform combines the transforms for the different modalities of Llama 4. It
    is made up of the following transforms:
    - :class:`torchtune.models.llama4.Llama4Tokenizer`
    - :class:`torchtune.models.clip.CLIPImageTransform`

    This transform can be used as a drop-in replacement for tokenizers in recipes and generation
    but handles additional transformations from the ``__call__`` method.

    Args:
        path (str): Path to pretrained tiktoken tokenizer file.
        tile_size (int): Size of the tiles to divide the image into.
        patch_size (int): Size of the patches used in the CLIP vision tranformer model. This is
            used to calculate the number of image embeddings per image.
        max_num_tiles (int): Only used if possible_resolutions is NOT given.
            Maximum number of tiles to break an image into.
            This will be used to generate possible_resolutions,
            e.g. [(224, 224), (224, 448), (448, 224)] if ``max_num_tiles = 2`` and ``tile_size = 224``.
            Default 4.
        pixel_shuffle_scaling_factor (float): scaling factor for pixel shuffle. Default is 0.5. You must ensure this
            matches the pixel shuffle scaling factor used in :class:`~torchtune.models.llama4.Llama4VisionProjectionHead`
            if modified from default.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Llama3 special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        image_mean (Optional[List[float]]): Mean values of each channel, used for normalization.
        image_std (Optional[List[float]]): Standard deviations for each channel, used for normalization.
        dtype (torch.dtype): Data type of transformed image. Default torch.bfloat16.
        prompt_template (Optional[_TemplateType]): template used to format the messages based on their role. This is used
            to add structured text around the actual messages. The structured text is used in three scenarios:

            - Task-specific templates to gear models for a particular task that it will expect after training
            - Model-specific templates that are required whenever the model is prompted, such as the [INST]
              tags in Llama2 and in Mistral
            - Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

            The extra text will still get tokenized as normal text, not as special tokens. Default is None.

    Examples:
        >>> model_transform = Llama4VisionTransform("/path/to/tokenizer.model", tile_size=224, patch_size=14)
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
        self.tokenizer = Llama4Tokenizer(
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
    ) -> torch.Tensor:
        tiles, ar = self.clip_transform({"image": image}, inference=inference).values()
        num_tiles, *_ = tiles.shape

        # add thumbnail if there are multiple tiles
        if num_tiles > 1:
            thumbnail = self.thumbnail_transform(image)
            tiles = torch.cat((tiles, thumbnail.unsqueeze(0)), dim=0)
        return tiles, ar

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        return self.tokenizer.encode(text=text, add_bos=add_bos, add_eos=add_eos)

    def decode(
        self,
        token_ids: List[int],
        truncate_at_eos: bool = True,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode a list of token ids into a string.

        Args:
            token_ids (List[int]): The list of token ids.
            truncate_at_eos (bool): Whether to truncate the string at the end of
                sequence token. Default is True.
            skip_special_tokens (bool): Whether to show or skip special tokens in the decoded string.
                Default is True.

        Returns:
            str: The decoded string.
        """
        return self.tokenizer.decode(
            token_ids,
            truncate_at_eos=truncate_at_eos,
            skip_special_tokens=skip_special_tokens,
        )

    def tokenize_message(
        self,
        message: Message,
        add_start_tokens: bool = True,
        add_end_tokens: bool = True,
    ) -> List[int]:
        """
        Tokenize a message into a list of token ids.

        Args:
            message (Message): The message to tokenize.
            add_start_tokens (bool): Whether to prepend a tokenized header to the message.
            add_end_tokens (bool): Whether to append eot or eom id at the end of the message.

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
            add_end_tokens (bool): Wether to add the tokenizer's eos_id. Default True.

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
            inference (bool): Whether to run in inference mode. Default is True.

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
