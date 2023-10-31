# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import torch

from sentencepiece import SentencePieceProcessor

from torch.nn.utils.rnn import pad_sequence


def to_tensor(
    ids: Union[List[int], List[List[int]]],
    padding_value: Optional[int] = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Convert input to PyTorch Tensor.

    Args:
        ids (Union[List[int], List[List[int]]]): Input data.
        padding_value (Optional[int]): Padding value. Default: None.
        dtype (torch.dtype): Output tensor dtype. Default: torch.long.

    Returns:
        Converted ids as PyTorch Tensor.

    Raises:
        TypeError: If the input type is not supported.

    Example:
        >>> input = [[1, 2, 3], [4, 5]]
        >>> tensor = to_tensor(input)
        >>> print(tensor)
        tensor([[1, 2, 3],  [4, 5]])
    """
    # Utilize JIT to check Generics types
    if torch.jit.isinstance(ids, List[int]):
        return torch.tensor(ids, dtype=dtype)
    elif torch.jit.isinstance(ids, List[List[int]]):
        if padding_value is None:
            output = torch.tensor(ids, dtype=dtype)
            return output
        else:
            output = pad_sequence(
                [torch.tensor(id, dtype=dtype) for id in ids],
                batch_first=True,
                padding_value=padding_value,
            )
            return output
    else:
        raise TypeError(f"Input type '{type(ids)}' not supported.")


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
        # If creating a tensor, the padding value must be provided. Encoding will
        # not pad non-tensor output.
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
            return_as_tensor (bool): Whether to return the result as a tensor, defaults to False.
            tensor_dtype (torch.dtype): The dtype of the returned tensor, defaults to torch.long.
            num_threads (int): Number of processing threads used for encoding, defaults to -1.

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
            return to_tensor(
                encoded_text, padding_value=self.spm_model.pad_id(), dtype=tensor_dtype
            )
        return encoded_text

    def decode(self, ids: Union[List[int], List[List[int]], torch.Tensor]) -> List[str]:
        """Decode token IDs to strings.

        Args:
            ids (Union[List[int], List[List[int]], torch.Tensor]): The input tokens to be decoded.

        Returns:
            List[str]: The decoded text.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        decoded_ids = self.spm_model.decode(ids)
        if isinstance(decoded_ids, str):
            return [decoded_ids]
        return decoded_ids
