# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

from PIL.Image import Image

from torchtune.data import Message, truncate
from torchtune.modules.tokenizers import ModelTokenizer, TikTokenBaseTokenizer


CL100K_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa

SPECIAL_TOKENS = {
    "<|begin_of_text|>": 128000,
    "<|end_of_text|>": 128001,
    "<|reserved_special_token_0|>": 128002,
    "<|reserved_special_token_1|>": 128003,
    "<|finetune_right_pad_id|>": 128004,
    "<|step_id|>": 128005,
    "<|start_header_id|>": 128006,
    "<|end_header_id|>": 128007,
    "<|eom_id|>": 128008,
    "<|eot_id|>": 128009,
    "<|python_tag|>": 128010,
    "<|image|>": 128011,
    "<|video|>": 128012,
}

NUM_RESERVED_SPECIAL_TOKENS = 256

RESERVED_TOKENS = {
    f"<|reserved_special_token_{2 + i}|>": 128013 + i
    for i in range(NUM_RESERVED_SPECIAL_TOKENS - len(SPECIAL_TOKENS))
}

LLAMA3_SPECIAL_TOKENS = {**SPECIAL_TOKENS, **RESERVED_TOKENS}


class Llama3Tokenizer(ModelTokenizer):
    """
    tiktoken tokenizer configured with Llama3 Instruct's special tokens, as described in
    https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3

    Args:
        path (str): Path to pretrained tiktoken tokenizer file.
        special_tokens (Optional[Dict[str, int]]): mapping containing special text tokens and
            their registered token IDs. If left as None, this will be set to the canonical
            Llama3 special tokens.
    """

    def __init__(
        self,
        path: str,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        special_tokens = special_tokens or LLAMA3_SPECIAL_TOKENS

        # Encode BOS and EOS, define pad ID
        self.bos_id = self.special_tokens["<|begin_of_text|>"]
        self.eos_id = self.special_tokens["<|end_of_text|>"]
        self.pad_id = self.special_tokens["<|finetune_right_pad_id|>"]

        # Encode extra special tokens
        self.start_header_id = self.special_tokens["<|start_header_id|>"]
        self.end_header_id = self.special_tokens["<|end_header_id|>"]
        self.eot_id = self.special_tokens["<|eot_id|>"]

        self.eom_id = self.special_tokens["<|eom_id|>"]
        self.python_tag = self.special_tokens["<|python_tag|>"]

        # Media tokens
        self.image_id = self.special_tokens["<|image|>"]
        self.video_id = self.special_tokens["<|video|>"]

        # During generation, stop when either eos_id or eot_id is encountered
        self.stop_tokens = [self.eos_id, self.eot_id]

        self.tt_model = TikTokenBaseTokenizer(
            path=path,
            name="llama3_tiktoken",
            pattern=CL100K_PATTERN,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            special_tokens=self.special_tokens,
        )

    @property
    def base_vocab_size(self) -> int:
        return self.tt_model.base_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.tt_model.vocab_size

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        return self.tt_model.encode(text=text, add_bos=add_bos, add_eos=add_eos)

    def decode(
        self,
        token_ids: List[int],
        truncate_at_eos: bool = True,
    ) -> str:
        """
        Decode a list of token ids into a string.

        Args:
            token_ids (List[int]): The list of token ids.
            truncate_at_eos (bool): Whether to truncate the string at the end of
                sequence token. Default is True.

        Returns:
            str: The decoded string.
        """
        return self.tt_model.decode(token_ids, truncate_at_eos=truncate_at_eos)

    def tokenize_message(
        self, message: Message, tokenize_header: bool = False
    ) -> Dict[str, Any]:
        """
        Tokenize a message into a list of token ids.

        Args:
            message (Message): The message to tokenize.
            tokenize_header (bool): Whether to prepend a tokenized header to each message.

        Returns:
            Dict[str, Any]: "tokens" - The list of token ids. "images" - PIL Image
                if message contains an image
        """
        image = None
        if tokenize_header:
            tokenized_header = (
                [self.start_header_id]
                + self.encode(message.role.strip(), add_bos=False, add_eos=False)
                + [self.end_header_id]
                + self.encode("\n\n", add_bos=False, add_eos=False)
            )
        else:
            tokenized_header = []
        if isinstance(message.content, Image):
            tokenized_body = [self.image_id]
            image = message.content
        else:
            tokenized_body = self.encode(
                message.content.strip(), add_bos=False, add_eos=False
            )
        if message.ipython:
            tokenized_body = [self.python_tag] + tokenized_body
        tokenized_message = tokenized_header + tokenized_body
        if message.eot:
            tokenized_message = tokenized_message + [self.eot_id]
        else:
            tokenized_message = tokenized_message + [self.eom_id]
        return {"tokens": tokenized_message, "images": image}

    def tokenize_messages(
        self,
        messages: List[Message],
        max_seq_len: Optional[int] = None,
        tokenize_header: bool = True,
        add_eos: bool = True,
    ) -> Dict[str, Any]:
        """
        Tokenize a list of messages into a list of token ids and masks.

        Args:
            messages (List[Message]): The list of messages to tokenize.
            max_seq_len (Optional[int]): The maximum sequence length.
            tokenize_header (bool): Whether to prepend a tokenized header to each message.

        Returns:
            Dict[str, Any]: "tokens" - list of token int ids, "mask" - list of booleans
                to indicate which tokens should be excluded from loss calculation,
                "images" - list of PIL Images from the messages, if any
        """
        tokens = [self.bos_id]
        # bos and eos are always masked
        mask = [True]
        images = []
        for message in messages:
            tokenized_message_dict = self.tokenize_message(
                message, tokenize_header=tokenize_header
            )
            tokens = tokens + tokenized_message_dict["tokens"]
            mask = mask + ([message.masked] * len(tokenized_message_dict["tokens"]))
            images = (
                images + [tokenized_message_dict["images"]]
                if tokenized_message_dict["images"] is not None
                else images
            )
            if max_seq_len and len(tokens) >= max_seq_len:
                break
        if add_eos:
            tokens = tokens + [self.eos_id]
            mask = mask + [True]
        if max_seq_len:
            tokens = truncate(tokens, max_seq_len, self.eos_id)
            mask = truncate(mask, max_seq_len, True)

        return {"tokens": tokens, "mask": mask, "images": images}
