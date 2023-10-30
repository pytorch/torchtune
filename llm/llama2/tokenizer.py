# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch

from sentencepiece import SentencePieceProcessor


class Tokenizer:
    """A wrapper around SentencePieceProcessor that supports batching and custom encoding/decoding.

    Args:
        spm_model (SentencePieceProcessor): The SentencePiece model.
        vocab_size (int): The size of the vocabulary.
        bos_id (int): The ID of the beginning-of-sentence token.
        eos_id (int): The ID of the end-of-sentence token.
        pad_id (int): The ID of the padding token.

    Examples:
        >>> tokenizer = Tokenizer.from_file("/path/to/spm_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True, return_as_tensor=True)
        >>> print(tokenized_text)
        tensor([    0, 31587, 29644,   102])
        >>> detokenized_text = tokenizer.decode([0, 31587, 29644, 102])
        >>> print(detokenized_text)
        ["Hello world!"]

        # Batched encoding
        >>> tokenized_text = tokenizer.encode(["Hello world!", "How are you?"], add_bos=True, add_eos=False, return_as_tensor=True)
        >>> print(tokenized_text)
        tensor([[    0, 31587, 29644, 102],
                [    0, 31587, 29644]])

        # Batched decoding
        >>> detokenized_text = tokenizer.decode([[0, 31587, 29644, 102],
                                                 [0, 31587, 29644]])
        >>> print(detokenized_text)
        [['Hello world!'], ['How are you?']]

        # Initialize with custom SentencePieceModel
        >>> spm.SentencePieceTrainer.train('--input=botchan.txt --model_prefix=m --vocab_size=2000')
        >>> spm_model = spm.SentencePieceProcessor()
        >>> spm_model.Load('/tmp/m.model')
        >>> tokenizer = Tokenizer(spm_model, vocab_size=2000, bos_id=0, eos_id=1, pad_id=2)
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
    def from_file(cls, path: Union[str, Path]) -> "Tokenizer":
        spm = SentencePieceProcessor()
        spm.load(path)
        return cls(spm, sp.vocab_size, sp.bos_id(), sp.eos_id(), sp.pad_id())

    def encode(
        self,
        text: Union[str, List[int]],
        add_bos: bool,
        add_eos: bool,
        return_as_tensor: bool = True,
        num_threads: int = -1,
    ) -> Union[torch.LongTensor, List[int], List[List[int]]]:
        """Encode a string or list of integers into a list of token IDs.

        Args:
            text (Union[str, List[int]]): The input text to be encoded.
            add_bos (bool): Whether to prepend BOS to the input.
            add_eos (bool): Whether to append EOS to the input.
            return_as_tensor (bool): Whether to return the output as a tensor.
            num_threads (int): Number of processing threads used for encoding.

        Returns:
            Union[torch.LongTensor, List[int], List[List[int]]]: The encoded text.
        """
        encoded_text = self.spm_model.encode(
            text,
            add_bos=add_bos,
            add_eos=add_eos,
            out_type=int,
            num_threads=num_threads,
        )
        if return_as_tensor:
            return torch.LongTensor(encoded_text)
        return encoded_text

    def decode(
        self, ids: Union[List[int], List[List[int]], torch.LongTensor]
    ) -> Union[List[str], List[List[str]]]:
        """Decode a list of token IDs into a string or list of strings.

        Args:
            ids (Union[List[int], List[List[int]], torch.LongTensor]): The input tokens to be decoded.

        Returns:
            Union[List[str], List[List[str]]: The decoded text.
        """
        if isinstance(ids, torch.LongTensor):
            ids = ids.tolist()
        decoded_ids = self.spm_model.decode(ids)
        if isinstance(decoded_ids, str):
            return [decoded_ids]
        return decoded_ids
