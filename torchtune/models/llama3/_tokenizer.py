# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torchtune.data import Message, truncate
from torchtune.data.tokenizers import TikTokenEncoding, Tokenizer


CL100K_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa

PAD_ID = 0


class Llama3Tokenizer(Tokenizer):
    """
    tiktoken tokenizer configured with Llama3 Instruct's special tokens, as described in
    https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3

    Args:
        path (str): Path to pretrained tiktoken tokenizer file.
    """

    def __init__(
        self,
        path: str,
    ):
        all_special_tokens_with_ids = self._get_all_special_tokens_with_ids()
        self.tt_model = TikTokenEncoding(
            path=path,
            name="llama3_tiktoken",
            pattern=CL100K_PATTERN,
            special_tokens=all_special_tokens_with_ids,
        )

        # Encode BOS and EOS, define pad ID
        self.bos_id = all_special_tokens_with_ids["<|begin_of_text|>"]
        self.eos_id = all_special_tokens_with_ids["<|end_of_text|>"]
        self.pad_id = PAD_ID

        # Encode extra special tokens
        self.step_id = all_special_tokens_with_ids["<|step_id|>"]
        self.start_header_id = all_special_tokens_with_ids["<|start_header_id|>"]
        self.end_header_id = all_special_tokens_with_ids["<|end_header_id|>"]
        self.eom_id = all_special_tokens_with_ids["<|eom_id|>"]
        self.eot_id = all_special_tokens_with_ids["<|eot_id|>"]
        self.python_tag = all_special_tokens_with_ids["<|python_tag|>"]

        # During generation, stop when either eos_id or eot_id is encountered
        self.stop_tokens = [self.eos_id, self.eot_id]

    def _get_all_special_tokens_with_ids(self) -> Dict[str, int]:
        special_tokens_json_path = Path(__file__).parent / "_special_tokens.json"
        with open(special_tokens_json_path, "r") as f:
            all_special_tokens = json.load(f)
        return all_special_tokens

    @property
    def base_vocab_size(self) -> int:
        return self.tt_model.base_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.tt_model.vocab_size

    def encode(
        self,
        text: str,
        add_bos: bool,
        add_eos: bool,
    ) -> List[int]:
        tokens = []
        if add_bos:
            tokens = tokens + [self.bos_id]
        tokens = tokens + self.tt_model.encode(text)
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

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
        if truncate_at_eos:
            try:
                k = token_ids.index(self.eos_id)
            except ValueError:
                k = None
            if k:
                token_ids = token_ids[:k]
        token_ids = [token_id for token_id in token_ids if token_id != self.bos_id]
        return self.tt_model.decode(token_ids)

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
