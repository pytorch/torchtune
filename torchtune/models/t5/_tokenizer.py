# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List

from torchtune.modules.transforms.tokenizers._sentencepiece import (
    SentencePieceBaseTokenizer,
)


class T5Tokenizer(SentencePieceBaseTokenizer):
    """
    Text tokenizer for T5.

    Args:
        path (str): the path to the T5 sentencepiece tokenizer file
        max_seq_len (int): the context length
        truncate (bool): whether to truncate the token sequence when longer than max_seq_len
    """

    def __init__(self, path: str, max_seq_len: int = 512, truncate: bool = True):
        super().__init__(path)
        self.max_seq_len = max_seq_len
        self.truncate = truncate

    def encode(self, text: str) -> List[int]:
        """
        Given a string, return the encoded list of token ids.

        Args:
            text (str): The text to encode.

        Returns:
            List[int]: The encoded list of token ids.
        """
        tokens = super().encode(
            text,
            add_bos=False,
            add_eos=True,
            trim_leading_whitespace=False,
            prefix=None,
        )
        if len(tokens) > self.max_seq_len:
            assert self.truncate, (
                "Tokenized text is larger than the maximum sequence length but "
                "truncate is set to False."
            )
            tokens = tokens[: self.max_seq_len]
            tokens[-1] = self.eos_id
        return tokens

    def __call__(
        self, sample: Dict[str, Any], inference: bool = False
    ) -> Dict[str, Any]:
        """
        Tokenize the "text" field in the sample.

        Args:
            sample (Dict[str, Any]): A sample with a "text" field containing a string to tokenize
            inference (bool): Unused by this tokenizer

        Returns:
            Dict[str, Any]: The sample with added "tokens" field and the "messages" field removed.
        """
        text = sample.pop("text")
        sample["tokens"] = self.encode(text)
        return sample
