# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from sentencepiece import SentencePieceProcessor


class Tokenizer:
    """A wrapper around SentencePieceProcessor.

    Args:
        spm_model (SentencePieceProcessor): The SentencePiece model.
        vocab_size (int): The size of the vocabulary.
        bos_id (int): The ID of the beginning-of-sentence token.
        eos_id (int): The ID of the end-of-sentence token.
        pad_id (int): The ID of the padding token.
        max_len (Optional[int]): maximum length of encoded token ID list during truncation. Default none

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
        max_len: Optional[int] = None,
    ):
        self.spm_model = spm_model
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_len = max_len

    @classmethod
    def from_file(cls, path: str, max_len: Optional[int] = None) -> "Tokenizer":
        """Initialize a `Tokenizer` instance from a SentencePiece model file.

        Args:
            path (str): The path to the SentencePiece model file.
            max_len (Optional[int]): See `Tokenizer.__init__` for details.

        Returns:
            Tokenizer: A `Tokenizer` instance.
        """
        spm = SentencePieceProcessor()
        spm.load(path)
        return cls(
            spm, spm.vocab_size(), spm.bos_id(), spm.eos_id(), spm.pad_id(), max_len
        )

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        truncate: bool = False,
    ) -> List[int]:
        """Encode text into token IDs.

        Args:
            text (str): The input text to be encoded, unbatched.
            add_bos (bool): Whether to prepend BOS to the input, defaults to True.
            add_eos (bool): Whether to append EOS to the input, defaults to True.
            truncate (bool): Whether to truncate the output on right, defaults to False.

        Returns:
            List[int]: The encoded token IDs.
        """
        assert type(text) == str, f"Expected string but got {type(text)}"
        tokens = self.spm_model.encode(
            text,
            add_bos=add_bos,
            add_eos=add_eos,
            out_type=int,
        )
        if truncate and self.max_len is not None:
            tokens = tokens[: self.max_len]
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to strings.

        Args:
            ids (List[int]): The input token IDs to be decoded.

        Returns:
            str: The decoded text.
        """
        return self.spm_model.decode(ids)
