# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

from torchtune.data import Message, truncate
from torchtune.data.tokenizers import SentencePieceEncoding, Tokenizer

WHITESPACE_CHARS = [" ", "\n", "\t", "\r", "\v"]


class MistralTokenizer(Tokenizer):
    """
    Mistral's implementation of the SentencePiece tokenizer

    Args:
        path (str): Path to pretrained tokenizer file.

    Example:
        # Accepts only non-batched input for now
        >>> tokenizer = SentencePieceEncoding("/path/to/spm_model")
        >>> tokenized_text = SentencePieceEncoding.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
    ):
        self._spm_model = SentencePieceEncoding(path)

        # Original tokenizer has no pad_id, which causes indexing errors when batch training
        self._spm_model.pad_id = 0

        # This is used in tokenize_messages: if the tokenizer does not
        # encode whitespace, then we can more easily split strings
        # on whitespace characters and encode them separately.
        self.encodes_whitespace = any(
            [self._spm_model.encode(c) for c in WHITESPACE_CHARS]
        )

        # During generation, stop when eos_id is encountered
        self.stop_tokens = [self.eos_id]

    @property
    def eos_id(self):
        return self._spm_model.eos_id

    @property
    def bos_id(self):
        return self._spm_model.bos_id

    @property
    def pad_id(self):
        return self._spm_model.pad_id

    @property
    def vocab_size(self):
        return self._spm_model.vocab_size

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

    def decode(
        self,
        token_ids: List[int],
        include_special: bool = False,
    ) -> str:
        return self._spm_model.decode(token_ids, include_special=include_special)

    def tokenize_messages(
        self, messages: List[Message], max_seq_len: Optional[int] = None
    ) -> Tuple[List[int], List[bool]]:
        r"""Tokenize a list of messages one at a time then concatenate them,
        returning a list of tokens and a list of masks.

        Note: llama2 sentencepiece has problems where in general
        encode(s1 + s2) != encode(s1) + encode(s2) due to whitespace handling.
        We can get around this by prepending s2 with a known token and slicing the
        beginning off the tokenized s2.

        Example:
            >>> tokenizer = SentencePieceEncoding(tokenizer_path)
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

        Returns:
            Tuple[List[int], List[bool]]: The tokenized messages
        """
        start_of_turn = True
        end_of_turn = False
        prev_ends_with_space = False
        tokenized_messages = []
        mask = []
        for message in messages:
            # If assistant message, this is the end of a turn
            end_of_turn = message.role == "assistant"

            # Prepend BOS on start of new turns
            if start_of_turn:
                tokenized_messages.append(self.bos_id)
                mask.append(message.masked)

            # We want to trim leading whitespace on the next message when
            # (a) it is a continuation of the turn (i.e. not the first message)
            # (b) the vocabulary explicitly encodes whitespace characters, and
            # (c) the previous message did not end with a space
            trim_leading_whitespace = (
                (not start_of_turn)
                and self.encodes_whitespace
                and not prev_ends_with_space
            )

            # Tokenize current message, append with masks
            tokens = self.encode(
                message.content.rstrip(" "),
                add_bos=False,
                add_eos=False,
                trim_leading_whitespace=trim_leading_whitespace,
            )
            prev_ends_with_space = message.content.endswith(" ")
            tokenized_messages.extend(tokens)
            mask.extend([message.masked] * len(tokens))

            # If assistant message, append EOS at end
            if end_of_turn:
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
        if max_seq_len:
            tokenized_messages = truncate(tokenized_messages, max_seq_len, self.eos_id)
            mask = truncate(mask, max_seq_len, message.masked)

        return tokenized_messages, mask
