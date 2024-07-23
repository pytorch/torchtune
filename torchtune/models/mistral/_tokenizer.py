# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping, Optional, Tuple

from torchtune.data import Message
from torchtune.modules.tokenizers import (
    ModelTokenizer,
    SentencePieceBaseTokenizer,
    tokenize_messages_no_special_tokens,
)
from torchtune.modules.transforms import Transform

WHITESPACE_CHARS = [" ", "\n", "\t", "\r", "\v"]


class MistralTokenizer(ModelTokenizer, Transform):
    """
    Mistral's implementation of the SentencePiece tokenizer

    Args:
        path (str): Path to pretrained tokenizer file.

    Examples:
        >>> tokenizer = MistralTokenizer("/path/to/spm_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
    ):
        self._spm_model = SentencePieceBaseTokenizer(path)

        # Original tokenizer has no pad_id, which causes indexing errors when batch training
        self._spm_model.pad_id = 0

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
        """
        Encode a string into a list of token IDs

        Args:
            text (str): The input text to be encoded, unbatched.
            add_bos (bool): Whether to prepend BOS special token (Beginning of Sentence) to the input, defaults to True.
            add_eos (bool): Whether to append EOS special token (End of Sentence) to the input, defaults to True.
            trim_leading_whitespace (bool): Whether to trim leading whitespace from
                underlying sentencepiece tokenization. Sentencepiece normally prepends
                whitespace to any tokenized text, which can cause differences where
                encode(s1) + encode(s2) != encode(s1 + s2) due to leading whitespace
                added to s2. Default: False
        Returns:
            List[int]: The encoded token IDs.
        """
        return self._spm_model.encode(
            text,
            add_bos=add_bos,
            add_eos=add_eos,
            trim_leading_whitespace=trim_leading_whitespace,
        )

    def decode(
        self,
        token_ids: List[int],
    ) -> str:
        """Decode token IDs to strings.

        Args:
            ids (List[int]): The input token IDs to be decoded.

        Returns:
            str: The decoded text.
        """
        return self._spm_model.decode(token_ids)

    def tokenize_messages(
        self, messages: List[Message], max_seq_len: Optional[int] = None
    ) -> Tuple[List[int], List[bool]]:
        r"""Tokenize a list of messages one at a time then concatenate them,
        returning a list of tokens and a list of masks.

        Note:
            sentencepiece has problems where in general
            encode(s1 + s2) != encode(s1) + encode(s2) due to whitespace handling.
            We can get around this by prepending s2 with a known token and slicing the
            beginning off the tokenized s2.

        Example:
            >>> tokenizer = MistralTokenizer(tokenizer_path)
            >>> messages = [
                Message(role="system", content="system message\n", masked=True),
                Message(role="user", content="user prompt\n", masked=True),
                Message(role="assistant", content="assistant response\n"),
            ]

            >>> # tokenize_messages encodes messages separately and concats
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

        Returns:
            Tuple[List[int], List[bool]]: The tokenized messages
        """
        return tokenize_messages_no_special_tokens(
            tokenizer=self,
            messages=messages,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            max_seq_len=max_seq_len,
        )

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Apply ``tokenize_messages`` to the "messages" field in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with a "messages" field containing
                a List[Message] to tokenize

        Returns:
            Mapping[str, Any]: The sample with added "tokens" and "mask" fields
                and the "messages" field removed.
        """
        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample
