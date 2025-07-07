# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from torchtune.data import ChatMLTemplate, Message, PromptTemplate, truncate
from torchtune.models.qwen2._tokenizer import (
    DEFAULT_QWEN2_TOKENIZER_BPE_CACHE_SIZE,
    ENDOFTEXT,
    IM_END,
)

from torchtune.models.qwen2_5._tokenizer import QWEN2_5_SPECIAL_TOKENS, Qwen2_5Tokenizer


class Qwen25VLTokenizer(Qwen2_5Tokenizer):
    """
    This class constructs a Qwen2.5-VL tokenizer, inheriting from Qwen2_5Tokenizer.

    This class overrides the tokenize_messages method to support vision tokens.

    See Qwen2_5Tokenizer for more details.
    """

    def __init__(
        self,
        path: str,
        merges_file: str,
        special_tokens: dict[str, int] = QWEN2_5_SPECIAL_TOKENS,
        max_seq_len: Optional[int] = None,
        *,
        prompt_template: Optional[PromptTemplate] = None,
        errors: str = "replace",
        unk_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: str = IM_END,
        pad_token: Optional[str] = ENDOFTEXT,
        bpe_cache_size: int = DEFAULT_QWEN2_TOKENIZER_BPE_CACHE_SIZE,
        truncation_type: str = "right",
    ):
        super().__init__(
            path=path,
            merges_file=merges_file,
            special_tokens=special_tokens,
            max_seq_len=max_seq_len,
            prompt_template=prompt_template,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            bpe_cache_size=bpe_cache_size,
            truncation_type=truncation_type,
        )

        self.im_start_id = self.special_tokens["<|im_start|>"]
        self.im_end_id = self.special_tokens["<|im_end|>"]
        self.image_pad_id = self.special_tokens["<|image_pad|>"]
        self.video_pad_id = self.special_tokens["<|video_pad|>"]
        self.vision_start_token_id = self.special_tokens["<|vision_start|>"]
        self.vision_end_token_id = self.special_tokens["<|vision_end|>"]

    def tokenize_messages(
        self,
        messages: list[Message],
        *,
        add_eos: bool = True,
    ) -> tuple[list[int], list[bool]]:
        """
        Given a list of messages, return a list of tokens for the concatenated
        and formatted messages.

        Args:
            messages (list[Message]): The message list to tokenize.
            add_eos (bool): Wether to add the tokenizer's eos_id at the end of the
                sequence of messages. Default is True.

        Returns:
            tuple[list[int], list[bool]]: The list of token ids and the list of masks.

        Raises:
            RuntimeError: If a message contains non-text content
        """
        assert not isinstance(self.prompt_template, ChatMLTemplate), (
            "Using ChatMLTemplate with tokenize_messages will result in multiple <|im_*|> tokens wrapping each message."
            "Please use a different template or set to None."
        )
        templated_messages = (
            self.prompt_template(messages)
            if self.prompt_template is not None
            else messages
        )

        tokenized_messages = []
        mask = []
        for i, message in enumerate(templated_messages):
            # message header
            tokens = self._tokenize_header(templated_messages, i)

            # message content
            for item in message.content:
                if item["type"] == "text":
                    tokens.extend(
                        self.encode(
                            item["content"],
                            add_bos=False,
                            add_eos=False,
                        )
                    )
                elif item["type"] == "image":
                    num_image_tokens = item.get("num_image_tokens")

                    tokens.append(self.vision_start_token_id)
                    tokens.extend([self.image_pad_id] * num_image_tokens)
                    tokens.append(self.vision_end_token_id)
                elif item["type"] == "video":
                    num_video_tokens = item.get("num_video_tokens")

                    tokens.append(self.vision_start_token_id)
                    tokens.extend([self.video_pad_id] * num_video_tokens)
                    tokens.append(self.vision_end_token_id)
                else:
                    raise RuntimeError(
                        f"Unsupported message content type: {item['type']}"
                    )

            # message footer
            tokens.extend(self._tokenize_footer(templated_messages, i))

            tokenized_messages.extend(tokens)
            mask.extend([message.masked] * len(tokens))

            # Break out early if we reach max_seq_len
            if self.max_seq_len and len(tokenized_messages) >= self.max_seq_len:
                break

        # Add the End-Of-Sequence token
        if add_eos:
            tokenized_messages.append(self.eos_id)
            mask.append(mask[-1])

        # Finally, truncate if necessary
        if self.max_seq_len:
            tokenized_messages = truncate(
                tokens=tokenized_messages,
                max_seq_len=self.max_seq_len,
                eos_id=self.eos_id if add_eos else None,
                truncation_type=self.truncation_type,
            )
            mask = truncate(
                tokens=mask,
                max_seq_len=self.max_seq_len,
                eos_id=True if add_eos else None,
                truncation_type=self.truncation_type,
            )

        return tokenized_messages, mask
