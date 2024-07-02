# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

from torchtune.data import Message, truncate
from torchtune.modules.tokenizers import ModelTokenizer, TikTokenBaseTokenizer


CL100K_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa

PAD_ID = 0

LLAMA3_SPECIAL_TOKENS = {
    "<|begin_of_text|>": 128000,
    "<|end_of_text|>": 128001,
    "<|start_header_id|>": 128006,
    "<|end_header_id|>": 128007,
    "<|eot_id|>": 128009,
    "<|eom_id|>": 128008,
    "<|python_tag|>": 128255,
}


class Llama3Tokenizer(ModelTokenizer):
    """
    tiktoken tokenizer configured with Llama3 Instruct's special tokens, as described in
    https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3

    Args:
        path (str): Path to pretrained tiktoken tokenizer file.
        special_tokens (Optional[Dict[str, int]]): mapping containing special text tokens and
            their registered token IDs. If left as None, this will be set to the canonical
            Llama3 special tokens.

    Examples:
        >>> tokenizer = Llama3Tokenizer("/path/to/tt_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        self.special_tokens = (
            special_tokens if special_tokens is not None else LLAMA3_SPECIAL_TOKENS
        )

        self._validate_special_tokens()

        # Encode BOS and EOS, define pad ID
        self.bos_id = self.special_tokens["<|begin_of_text|>"]
        self.eos_id = self.special_tokens["<|end_of_text|>"]
        self.pad_id = PAD_ID

        # Encode extra special tokens
        self.start_header_id = self.special_tokens["<|start_header_id|>"]
        self.end_header_id = self.special_tokens["<|end_header_id|>"]
        self.eot_id = self.special_tokens["<|eot_id|>"]

        self.eom_id = self.special_tokens["<|eom_id|>"]
        self.python_tag = self.special_tokens["<|python_tag|>"]

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

    def _validate_special_tokens(
        self,
    ):
        """
        Validate that required special tokens are passed into the tokenizer. The
        following special tokens are required: <|begin_of_text|>, <|end_of_text|>,
        <|start_header_id|>, <|end_header_id|>, <|eot_id|>, <|eom_id|>, <|python_tag|>
        """
        for token in LLAMA3_SPECIAL_TOKENS.keys():
            if token not in self.special_tokens:
                raise ValueError(f"{token} missing from special_tokens")

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
    ) -> List[int]:
        """
        Tokenize a message into a list of token ids.

        Args:
            message (Message): The message to tokenize.
            tokenize_header (bool): Whether to prepend a tokenized header to each message.

        Returns:
            List[int]: The list of token ids.
        """
        if tokenize_header:
            tokenized_header = (
                [self.start_header_id]
                + self.encode(message.role.strip(), add_bos=False, add_eos=False)
                + [self.end_header_id]
                + self.encode("\n\n", add_bos=False, add_eos=False)
            )
        else:
            tokenized_header = []
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
        return tokenized_message

    def tokenize_messages(
        self,
        messages: List[Message],
        max_seq_len: Optional[int] = None,
        tokenize_header: bool = True,
        add_eos: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Tokenize a list of messages into a list of token ids and masks.

        Args:
            messages (List[Message]): The list of messages to tokenize.
            max_seq_len (Optional[int]): The maximum sequence length.
            tokenize_header (bool): Whether to prepend a tokenized header to each message.

        Returns:
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.
        """
        tokens = [self.bos_id]
        # bos and eos are always masked
        mask = [True]
        for message in messages:
            tokenized_message = self.tokenize_message(
                message, tokenize_header=tokenize_header
            )
            tokens = tokens + tokenized_message
            mask = mask + ([message.masked] * len(tokenized_message))
            if max_seq_len and len(tokens) >= max_seq_len:
                break
        if add_eos:
            tokens = tokens + [self.eos_id]
            mask = mask + [True]
        if max_seq_len:
            tokens = truncate(tokens, max_seq_len, self.eos_id)
            mask = truncate(mask, max_seq_len, True)
        return tokens, mask
