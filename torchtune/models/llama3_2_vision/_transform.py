# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Mapping, Optional, Tuple

from torchtune.data import Message, PromptTemplate

from torchtune.models.clip import CLIPImageTransform
from torchtune.models.llama3 import llama3_tokenizer
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform, VisionCrossAttentionMask


class Llama3VisionTransform(ModelTokenizer, Transform):
    """
    This transform combines the transforms for the different modalities of Llama 3.2 Vision. It
    is made up of the following transforms:
    - :class:`torchtune.models.llama3.Llama3Tokenizer`
    - :class:`torchtune.models.clip.CLIPImageTransform`
    - :class:`torchtune.modules.transforms.VisionCrossAttentionMask`

    This transform can be used as a drop-in replacement for tokenizers in recipes and generation
    but handles additional transformations from the `__call__` method.

    Args:
        path (str): Path to pretrained tiktoken tokenizer file.
        tile_size (int): Size of the tiles to divide the image into.
        patch_size (int): Size of the patches used in the CLIP vision tranformer model. This is
            used to calculate the number of image embeddings per image.
        max_num_tiles (int): Only used if possible_resolutions is NOT given.
            Maximum number of tiles to break an image into.
            This will be used to generate possible_resolutions,
            e.g. [(224, 224), (224, 448), (448, 224)] if max_num_tiles = 2 and tile_size = 224.
            Default 4.
        special_tokens (Optional[Dict[str, int]]): mapping containing special text tokens and
            their registered token IDs. If left as None, this will be set to the canonical
            Llama3 special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        image_mean (Optional[Tuple[float, float, float]]): Mean values of each channel, used for normalization.
        image_std (Optional[Tuple[float, float, float]]): Standard deviations for each channel, used for normalization.
        prompt_template (Optional[PromptTemplate]): template used to format the messages based on their role. This is used
            to add structured text around the actual messages. The structured text is used in three scenarios:

            - Task-specific templates to gear models for a particular task that it will expect after training
            - Model-specific templates that are required whenever the model is prompted, such as the [INST]
              tags in Llama2 and in Mistral
            - Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

            The extra text will still get tokenized as normal text, not as special tokens. Default is None.

    Examples:
        >>> model_transform = Llama3VisionTransform("/path/to/tokenizer.model", tile_size=224, patch_size=14)
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
        special_tokens: Optional[Dict[str, int]] = None,
        max_seq_len: Optional[int] = None,
        image_mean: Optional[Tuple[float, float, float]] = None,
        image_std: Optional[Tuple[float, float, float]] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        self.tokenizer = llama3_tokenizer(
            path,
            special_tokens_path=special_tokens,
            max_seq_len=max_seq_len,
            prompt_template=prompt_template,
        )
        self.transform_image = CLIPImageTransform(
            image_mean=image_mean,
            image_std=image_std,
            tile_size=tile_size,
            possible_resolutions=None,
            max_num_tiles=max_num_tiles,
            resample="bilinear",
            resize_to_max_canvas=False,
        )
        self.xattn_mask = VisionCrossAttentionMask(
            tile_size=tile_size,
            patch_size=patch_size,
            image_token_id=self.tokenizer.image_id,
            max_num_tiles=max_num_tiles,
        )

        self.stop_tokens = self.tokenizer.stop_tokens
        self.max_seq_len = max_seq_len
        self.image_seq_len = max_num_tiles * (self.xattn_mask.patches_per_tile + 1)
        self.prompt_template = prompt_template
        self.pad_id = self.tokenizer.pad_id

    @property
    def base_vocab_size(self) -> int:
        return self.tokenizer.base_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

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
        tokenize_header: bool = True,
        tokenize_end: bool = True,
    ) -> List[int]:
        """
        Tokenize a message into a list of token ids.

        Args:
            message (Message): The message to tokenize.
            tokenize_header (bool): Whether to prepend a tokenized header to the message.
            tokenize_end (bool): Whether to append eot or eom id at the end of the message.

        Returns:
            List[int]: The list of token ids.
        """
        return self.tokenizer.tokenize_message(
            message=message,
            tokenize_header=tokenize_header,
            tokenize_end=tokenize_end,
        )

    def tokenize_messages(
        self,
        messages: List[Message],
        add_eos: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Tokenize a list of messages into a list of token ids and masks.

        Args:
            messages (List[Message]): The list of messages to tokenize.
            add_eos (bool): Wether to add the tokenizer's eos_id. Default True.

        Returns:
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.
        """
        return self.tokenizer.tokenize_messages(
            messages=messages,
            add_eos=add_eos,
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
                - encoder_mask: List[bool] of masks for the transformed images
        """
        encoder_input = {"images": [], "aspect_ratio": []}
        messages = sample["messages"]
        for message in messages:
            for image in message.get_media():
                out = self.transform_image({"image": image}, inference=inference)
                encoder_input["images"].append(out["image"])
                encoder_input["aspect_ratio"].append(out["aspect_ratio"])

        sample["encoder_input"] = encoder_input
        sample = self.tokenizer(sample, inference=inference)
        sample = self.xattn_mask(sample, inference=inference)
        return sample
