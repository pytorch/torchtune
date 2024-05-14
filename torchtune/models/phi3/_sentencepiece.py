# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

from sentencepiece import SentencePieceProcessor
from torchtune.data._types import Message
from torchtune.data._utils import truncate


class Phi3MiniSentencePieceTokenizer:
    """A wrapper around SentencePieceProcessor.

    Args:
        path (str): Path to pretrained tokenizer file.

    Example:
        # Accepts only non-batched input for now
        >>> tokenizer = SentencePieceTokenizer("/path/to/spm_model")
        >>> tokenized_text = SentencePieceTokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
    ):
        spm_model = SentencePieceProcessor()
        spm_model.load(path)
        self.spm_model = spm_model

        self.special_tokens = {
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

        self.vocab_size = spm_model.vocab_size()
        self.bos_id = spm_model.bos_id()
        self.eos_id = self.special_tokens["<|endoftext|>"]
        self.pad_id = self.special_tokens["<|endoftext|>"]

        # During generation, stop when eos_id is encountered
        self.stop_tokens = [self.eos_id]

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        trim_leading_whitespace: bool = False,
        prefix: Optional[str] = None,
    ) -> List[int]:
        """Encode text into token IDs.

        Args:
            text (str): The input text to be encoded, unbatched.
            add_bos (bool): Whether to prepend BOS to the input, defaults to True.
            add_eos (bool): Whether to append EOS to the input, defaults to True.
            trim_leading_whitespace (bool): Whether to trim leading whitespace from
                underlying sentencepiece tokenization. Sentencepiece normally prepends
                whitespace to any tokenized text, which can cause differences where
                encode(s1) + encode(s2) != encode(s1 + s2) due to leading whitespace
                added to s2. Default: False
            prefix (Optional[str]): Optional string to encode for trimming leading
                whitespaces. Used only if trim_leading_whitespace=True. Default: None
        Returns:
            List[int]: The encoded token IDs.
        """
        if trim_leading_whitespace:
            # Can define our own custom prefix depending on vocab if needed
            if not hasattr(self, "prefix"):
                self.prefix = prefix or "\n"
                self.encoded_prefix = self.spm_model.encode(
                    self.prefix, add_bos=False, add_eos=False
                )
            start_idx = len(self.encoded_prefix) + int(add_bos)
            return self.spm_model.encode(
                self.prefix + text,
                add_bos=add_bos,
                add_eos=add_eos,
                out_type=int,
            )[start_idx:]
        else:
            return self.spm_model.encode(
                text,
                add_bos=add_bos,
                add_eos=add_eos,
                out_type=int,
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
            if token_id in self.special_tokens.values():
                continue
            else:
                ids_for_decode.append(token_id)
        return self.spm_model.decode(ids_for_decode)

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
            >>> tokenizer = SentencePieceTokenizer(tokenizer_path)
            >>> messages = [
                Message(role="system", content="system message\n", masked=True),
                Message(role="user", content="user prompt\n", masked=True),
                Message(role="assistant", content="assistant response\n"),
                ]
            # tokenize_messages encodes messages separately and concats
            >>> tokenizer.tokenize_messages(messages, max_seq_len)[0]
            [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]


            # Same result as encoding the full string in one go
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
        prev_ends_with_space = False
        tokenized_messages = []
        mask = []

        # The chat template in HF adds a bunch of newlines
        new_line_token_id = self.encode("\n", add_bos=False, add_eos=False)

        for message in messages:
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
                if ignore_system_prompts:
                    continue
                else:
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
            prev_ends_with_space = message.content.endswith(" ")
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
