# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional
import torch
from torch.nn.utils.rnn import pad_sequence
from sentencepiece import SentencePieceProcessor


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
    ) -> List[int]:
        """Encode text into token IDs.

        Args:
            text (str): The input text to be encoded, unbatched.
            add_bos (bool): Whether to prepend BOS to the input, defaults to True.
            add_eos (bool): Whether to append EOS to the input, defaults to True.

        Returns:
            List[int]: The encoded token IDs.
        """
        assert type(text) == str, f"Expected string but got {type(text)}"
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

    def to_tensor(self, tokens: List[List[int]], pad_value: int, max_length: Optional[int] = None) -> torch.LongTensor:
        """Converts tokens to a LongTensor.

        Args:
            tokens (List[List[int]]): The input tokens.
            max_length (Optional[int]): Maximum length of the output tensor, defaults to None.

        Returns:
            Tensorized tokens.

        TODO: Clean this up, remove double padding
        """
        padded_tokens = pad_sequence(
            [torch.tensor(t, dtype=torch.long) for t in tokens],
            batch_first=True,
            padding_value=pad_value,
        )
        if max_length is not None:
            if padded_tokens.shape[-1] > max_length:
                padded_tokens = padded_tokens[:, :max_length]
        return padded_tokens
