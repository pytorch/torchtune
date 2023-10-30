# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import List, Optional, Union

import torch

from sentencepiece import SentencePieceProcessor

def to_tensor(ids: Any, padding_value: Optional[int] = None, dtype: torch.dtype = torch.long) -> Tensor:
    """Convert input to PyTorch Tensor.

    Args:
        input (Any): Input data.
        padding_value (Optional[int]): Padding value.
        dtype (torch.dtype): Output tensor dtype.

    Returns:
        Tensor: Converted tensor.

    Raises:
        TypeError: If the input type is not supported.

    Example:
        >>> input = [[1, 2, 3], [4, 5]]
        >>> tensor = to_tensor(input)
        >>> print(tensor)
        tensor([[1, 2, 3],  [4, 5]])
    """
    if isinstance(ids, List[int]):
        return torch.tensor(input, dtype=torch.long)
    elif torch.jit.isinstance(input, List[List[int]]):
        if padding_value is None:
            output = torch.tensor(input, dtype=dtype)
            return output
        else:
            output = pad_sequence(
                [torch.tensor(ids, dtype=dtype) for ids in input], batch_first=True, padding_value=float(padding_value)
            )
            return output
    else:
        raise TypeError(f"Input type '{type(input)}' not supported.")


class Tokenizer:
    """A wrapper around SentencePieceProcessor that supports batching and custom encoding/decoding.

    Args:
        spm_model (SentencePieceProcessor): The SentencePiece model.
        vocab_size (int): The size of the vocabulary.
        bos_id (int): The ID of the beginning-of-sentence token.
        eos_id (int): The ID of the end-of-sentence token.
        pad_id (int): The ID of the padding token.

    Example:
        >>> tokenizer = Tokenizer.from_file("/path/to/spm_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True, return_as_tensor_with_dtype=torch.long)
        >>> print(tokenized_text)
        tensor([    0, 31587, 29644,   102])
        >>> detokenized_text = tokenizer.decode([0, 31587, 29644, 102])
        >>> print(detokenized_text)
        ["Hello world!"]

        # Batched encoding
        >>> tokenized_text = tokenizer.encode(
            ["Hello world!", "How are you?"],
            add_bos=True,
            add_eos=False,
            return_as_tensor_with_dtype=torch.long
        )
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
    def from_file(cls, path: str) -> "Tokenizer":
        """Initialize a `Tokenizer` instance from a SentencePiece model file.

        Args:
            path (str): The path to the SentencePiece model file.

        Returns:
            Tokenizer: A `Tokenizer` instance.
        """
        spm = SentencePieceProcessor()
        spm.load(path)
        return cls(spm, spm.vocab_size, spm.bos_id(), spm.eos_id(), spm.pad_id())

    def encode(
        self,
        text: Union[str, List[int]],
        add_bos: bool = True,
        add_eos: bool = True,
        return_as_tensor: bool = False,
        tensor_dtype: torch.dtype = torch.long,
        num_threads: int = -1,
    ) -> Union[torch.Tensor, List[int], List[List[int]]]:
        """Encode string(s) into token IDs.

        Args:
            text (Union[str, List[int]]): The input text to be encoded.
            add_bos (bool): Whether to prepend BOS to the input, defaults to True.
            add_eos (bool): Whether to append EOS to the input, defaults to True.
            return_as_tensor_with_dtype (torch.dtype): The dtype of the returned tensor, defaults to torch.long.
            num_threads (int): Number of processing threads used for encoding.

        Returns:
            Union[torch.Tensor, List[int], List[List[int]]]: The encoded text.
        """
        encoded_text = self.spm_model.encode(
            text,
            add_bos=add_bos,
            add_eos=add_eos,
            out_type=int,
            num_threads=num_threads,
        )
        if return_as_tensor:
            return to_tensor(encoded_text, padding_value = self.spm_model.pad_id(), dtype = tensor_dtype)
        return encoded_text

    def decode(
        self, ids: Union[List[int], List[List[int]], torch.Tensor]
    ) -> Union[List[str], List[List[str]]]:
        """Decode token IDs to strings.

        Args:
            ids (Union[List[int], List[List[int]], torch.Tensor]): The input tokens to be decoded.

        Returns:
            Union[List[str], List[List[str]]: The decoded text.
        """
        if isinstance(ids, torch.LongTensor):
            ids = ids.tolist()
        decoded_ids = self.spm_model.decode(ids)
        if isinstance(decoded_ids, str):
            return [decoded_ids]
        return decoded_ids
