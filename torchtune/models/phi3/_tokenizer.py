# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

from torchtune.data import Message, truncate
from torchtune.modules.tokenizers import ModelTokenizer, SentencePieceBaseTokenizer

PHI3_SPECIAL_TOKENS = {
    "<|endoftext|>": 32000,
    "<|assistant|>": 32001,
    "<|placeholder1|>": 32002,
    "<|placeholder2|>": 32003,
    "<|placeholder3|>": 32004,
    "<|placeholder4|>": 32005,
    "<|system|>": 32006,
    "<|end|>": 32007,
    "<|placeholder5|>": 32008,
    "<|placeholder6|>": 32009,
    "<|user|>": 32010,
}


class Phi3MiniTokenizer(ModelTokenizer):
    """
    SentencePiece tokenizer configured with Phi3 Mini's special tokens.

    Args:
        path (str): Path to pretrained tokenizer file.
        special_tokens (Optional[Dict[str, int]]): mapping containing special text tokens and
            their registered token IDs. If left as None, this will be set to the canonical
            Phi3 special tokens.

    Examples:
        >>> tokenizer = Phi3MiniTokenizer("/path/to/spm_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        self._spm_model = SentencePieceBaseTokenizer(path)

        self.special_tokens = (
            special_tokens if special_tokens is not None else PHI3_SPECIAL_TOKENS
        )

        # Use custom EOS and pad ids instead of SentencePiece's
        self.eos_id = self.special_tokens["<|endoftext|>"]
        self.pad_id = self.special_tokens["<|endoftext|>"]

        # During generation, stop when eos_id is encountered
        self.stop_tokens = [self.eos_id]

    @property
    def vocab_size(self):
        return self._spm_model.vocab_size

    @property
    def bos_id(self):
        return self._spm_model.bos_id

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        trim_leading_whitespace: bool = False,
    ) -> List[int]:
        return self._spm_model.encode(
            text,
            add_bos=add_bos,
            add_eos=add_eos,
            trim_leading_whitespace=trim_leading_whitespace,
        )

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to strings.

        Args:
            ids (List[int]): The input token IDs to be decoded.

        Returns:
            str: The decoded text.
        """
        ids_for_decode = []
        for token_id in ids:
            # Filter out special tokens and the placeholder tokens added
            # by the Phi3 team
            if token_id >= 32_000 and token_id <= 32_064:
                continue
            else:
                ids_for_decode.append(token_id)
        return self._spm_model.decode(ids_for_decode)

    def tokenize_messages(
        self,
        messages: List[Message],
        max_seq_len: Optional[int] = None,
        *,
        add_eos: bool = False,
        ignore_system_prompts: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        r"""Tokenize a list of messages one at a time then concatenate them,
        returning a list of tokens and a list of masks.

        Example:
            >>> tokenizer = Phi3MiniTokenizer(tokenizer_path)
            >>> messages = [
                Message(role="system", content="system message\n", masked=True),
                Message(role="user", content="user prompt\n", masked=True),
                Message(role="assistant", content="assistant response\n"),
            ]

            # tokenize_messages encodes messages separately and concats
            >>> tokenizer.tokenize_messages(messages, max_seq_len)[0]
            [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]

            >>> # Same result as encoding the full string in one go
            >>> tokenizer.encode(''.join([message.content for message in messages]))
            [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]


        Args:
            messages (List[Message]): A list of messages, each containing role, content,
                and masked attributes.
            max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
                Default: None
            add_eos (bool): Whether to append EOS after assistant message, default to False
            ignore_system_prompts (bool): Whether to ignore system prompts. This matches the HF implementation, default to True.

        Raises:
            ValueError: If the role is not "user", "assistant", or "system".

        Returns:
            Tuple[List[int], List[bool]]: The tokenized messages
        """
        start_of_turn = True
        end_of_turn = False
        tokenized_messages = []
        mask = []

        # The chat template in HF adds a bunch of newlines
        new_line_token_id = self.encode("\n", add_bos=False, add_eos=False)

        for message in messages:
            # Skip system prompt
            if ignore_system_prompts and message.role == "system":
                continue

            # Prepend BOS on start of new turns
            if start_of_turn:
                tokenized_messages.append(self.bos_id)
                mask.append(message.masked)

            # Add special tokens
            if message.role == "user":
                tokenized_messages.append(self.special_tokens["<|user|>"])
                mask.append(message.masked)
            elif message.role == "assistant":
                tokenized_messages.append(self.special_tokens["<|assistant|>"])
                # If assistant message, this is the end of a turn
                end_of_turn = True
                mask.append(message.masked)
            elif message.role == "system":
                tokenized_messages.append(self.special_tokens["<|system|>"])
                mask.append(message.masked)
            else:
                raise ValueError(
                    f"Unknown role '{message.role}' for message: '{message.content}'"
                )

            # Add new line token
            tokenized_messages.extend(new_line_token_id)
            mask.extend([message.masked] * len(new_line_token_id))

            # Tokenize current message, append with masks
            tokens = self.encode(
                message.content.rstrip(" "),
                add_bos=False,
                add_eos=False,
                trim_leading_whitespace=True,  # Always trim whitespace (just to match HF tokenizer implementation)
            )
            tokens = tokens + [self.special_tokens["<|end|>"]] + new_line_token_id
            tokenized_messages.extend(tokens)
            mask.extend([message.masked] * len(tokens))

            # If assistant message, append EOS at end
            if end_of_turn and add_eos:
                tokenized_messages.append(self.eos_id)
                mask.append(message.masked)
                end_of_turn = False
                start_of_turn = True
            else:
                start_of_turn = False

            # Break out early if we reach max_seq_len
            if max_seq_len and len(tokenized_messages) >= max_seq_len:
                break

        # Finally, truncate if necessary
        if max_seq_len and len(tokenized_messages) >= max_seq_len:
            tokenized_messages = truncate(tokenized_messages, max_seq_len, self.eos_id)
            mask = truncate(mask, max_seq_len, message.masked)

        return tokenized_messages, mask
