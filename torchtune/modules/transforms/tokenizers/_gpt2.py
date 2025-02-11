# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import re
from functools import lru_cache
from typing import List

from torchtune.modules.transforms.tokenizers._utils import BaseTokenizer


@lru_cache()
def bytes_to_unicode() -> dict:
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    Original paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

    This is standard implementation based on:
    https://github.com/huggingface/transformers/blob/6b550462139655d488d4c663086a63e98713c6b9/src/transformers/models/gpt2/tokenization_gpt2.py#L36
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: tuple) -> set:
    """
    Args:
        word (tuple): Word is represented as tuple of symbols (symbols being variable-length strings).

    Returns:
        set of symbol pairs in a word.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class GPT2BaseTokenizer(BaseTokenizer):
    """
    A lightweight version of the basic GPT2Tokenizer.

    Args:
        vocab_path (str): Path to vocab.json file.
        merges_path (str): Path to merges.txt file.
        unk_id (int): unkown token id.
        bos_id (int): beginning-of-sequence token id.
        eos_id (int): end-of-sequence token id.
        pad_id (int): padding token id.


    Examples:
        >>> tokenizer = GPT2BaseTokenizer("vocab.json", "merges.txt", "replace", 1, 1, 1, 1)
        >>> tokenized_text = tokenizer.encode("Hello world!")
        >>> print(tokenized_text)
        [1, 9906, 1917, 1]
    """

    def __init__(
        self,
        vocab_path: str,
        merges_path: str,
        unk_id: int = None,
        bos_id: int = None,
        eos_id: int = None,
        pad_id: int = None,
    ):
        with open(vocab_path, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_path, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        # We do not use external regex library, so this is slightly modified version of the original line
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\w]+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
            re.UNICODE,
        )

        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.unk_token = self.encoder.get(self.unk_id)

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def bpe(self, token: str) -> str:
        """
        Returns pair for the given token.

        Args:
            token (str): Passed token.

        Returns:
            Pair token for the given token.
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:
        """
        Tokenize, but not encode given text.

        Args:
            text (str): text to tokenize

        Returns:
            BPE Tokens.
        """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))

        return bpe_tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.unk_token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index)

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """
        Tokenize and encode given text.

        Args:
            text (str): text to encode.
            add_bos (bool): True if bos token must be added.
            add_eos (bool): True if eos token must be added.

        Returns:
            Tokenized and encoded text.
        """
        bpe_tokens = list(map(self._convert_token_to_id, self._tokenize(text)))

        if add_bos:
            bpe_tokens = [self.bos_id] + bpe_tokens
        if add_eos:
            bpe_tokens = bpe_tokens + [self.eos_id]
        return bpe_tokens

    def decode(self, tokens: list) -> List[str]:
        """
        Decode sequence of the given tokens into string.

        Args:
            tokens (list): List of the integers, which represent encoded tokens.

        Returns:
            Decoded text.
        """

        decoded_tokens = list(map(self._convert_id_to_token, tokens))
        return decoded_tokens
