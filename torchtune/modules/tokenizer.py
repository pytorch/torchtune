# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

from sentencepiece import SentencePieceProcessor
from torchtune.data._types import Message
from torchtune.data._utils import truncate


class Tokenizer:
    """A wrapper around SentencePieceProcessor.

    Args:
        spm_model (SentencePieceProcessor): The SentencePiece model.
        vocab_size (int): The size of the vocabulary.
        bos_id (int): The ID of the beginning-of-sentence token.
        eos_id (int): The ID of the end-of-sentence token.
        pad_id (int): The ID of the padding token.

    Example:
        # Accepts only non-batched input for now
        >>> tokenizer = Tokenizer.from_file("/path/to/spm_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        spm_model: SentencePieceProcessor,
        vocab_size: int,
        bos_id: int,
        eos_id: int,
        pad_id: int,
    ):
        self.spm_model = spm_model
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    @classmethod
    def from_file(cls, path: str) -> "Tokenizer":
        """Initialize a `Tokenizer` instance from a SentencePiece model file.

        Args:
            path (str): The path to the SentencePiece model file.

        Returns:
            Tokenizer: A `Tokenizer` instance.
        """
        spm = SentencePieceProcessor()
        spm.load(path)
        return cls(spm, spm.vocab_size(), spm.bos_id(), spm.eos_id(), spm.pad_id())

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        trim_leading_whitespace: bool = False,
    ) -> List[int]:
        """Encode text into token IDs.

        Args:
            text (str): The input text to be encoded, unbatched.
            add_bos (bool): Whether to prepend BOS to the input, defaults to True.
            add_eos (bool): Whether to append EOS to the input, defaults to True.
            trim_leading_whitespace (bool): Whether to trim leading whitespace from
                underlying sentencepiece tokenization. Default: False
        Returns:
            List[int]: The encoded token IDs.
        """
        if trim_leading_whitespace:
            return self.spm_model.encode(
                "\n" + text,
                add_bos=add_bos,
                add_eos=add_eos,
                out_type=int,
            )[2:]
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
        return self.spm_model.decode(ids)

    def tokenize_messages(
        self, messages: List[Message], max_seq_len: Optional[int] = None
    ) -> Tuple[List[int], List[bool]]:
        start_of_turn = True
        end_of_turn = False
        tokenized_messages = []
        mask = []
        for message in messages:
            # If assistant message, this is the end of a turn
            end_of_turn = message.role == "assistant"
            # Prepend BOS on start of new turns
            if start_of_turn:
                tokenized_messages.append(self.bos_id)
                mask.append(message.masked)
                # Tokenize current message, append with masks
            tokens = self.encode(
                message.content,
                add_bos=False,
                add_eos=False,
                trim_leading_whitespace=not start_of_turn,
            )
            start_of_turn = False
            tokenized_messages.extend(tokens)
            mask.extend([message.masked] * len(tokens))

            # Break out early if we reach max_seq_len
            if max_seq_len and len(tokens) >= max_seq_len:
                break

            # If assistant message, append EOS at end
            if end_of_turn:
                tokenized_messages.append(self.eos_id)
                mask.append(message.masked)
                end_of_turn = False
                start_of_turn = True

        # Finally, truncate if necessary
        if max_seq_len:
            tokenized_messages = truncate(tokenized_messages, max_seq_len, self.eos_id)
            mask = truncate(mask, max_seq_len, message.masked)

        return tokenized_messages, mask
